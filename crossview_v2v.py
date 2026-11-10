"""One real-video CrossView run. No model changes or object removal.

Warp follows the author's per-frame depth reprojection and painter splat,
retaining the existing full K (the author node assumes square pixels).
H3 runs in the already installed DiffSynth runtime, not ComfyUI itself.
"""
from pathlib import Path
import argparse
import datetime
import gc
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import time

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'outputs/p0_dynamic_v2v'
SOURCE = Path(os.environ.get('CROSSVIEW_SOURCE', ROOT/'inputs/source.mp4'))
OLD = ROOT/'outputs/lanechange_fast_20261031/inputs'
MODEL = Path(os.environ.get('CROSSVIEW_MODEL_ROOT', ROOT/'models/MiniMax-H3-Ref2VA-BF16'))
WEIGHTS = Path(os.environ.get('CROSSVIEW_WEIGHTS_ROOT', ROOT/'weights'))
N, FPS, W, H, SEED = 124, 24, 832, 480, 20261028
SHAPE = (1, 24, 37, 30, 52)
MODEL_FPS = 24  # Native Ref2VA video/audio position schedule.
PROMPT = 'crossview'
DEVICES = ['cuda:0', 'cuda:1']

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))

def state(phase, **values):
    record = dict(phase=phase, updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), **values)
    path = OUT/'run_state.json'; temp = path.with_suffix('.tmp')
    save_json(temp, record); temp.replace(path)
    print(json.dumps(record), flush=True)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read_video(path):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    assert abs(fps-FPS) < 1e-6 and len(frames) == N, (path, fps, len(frames))
    assert all(f.shape == (H, W, 3) for f in frames)
    return frames

def load_author():
    path = ROOT/'deps/ComfyUI-CrossViewWarp/crossview_warp_node.py'
    spec = importlib.util.spec_from_file_location('crossview_author_warp', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def warp_frame(rgb, depth, valid, K, relative_c2w, author):
    """Author's z-sorted painter / radius-2 splat, with original fx/fy/cx/cy.

    rgb/depth/valid are from the SAME current frame, never a frame-0 cloud.
    c2w is a target-view offset in that frame's OpenCV camera coordinates.
    """
    yy, xx = np.mgrid[:H, :W]
    z = np.where(valid, depth, np.nan).astype(np.float64)
    finite = np.isfinite(z) & (z > 0)
    finite &= z < (np.percentile(z[finite], 99.5) if finite.any() else 0)
    xyz = np.stack([(xx-K[0,2])/K[0,0]*z, (yy-K[1,2])/K[1,1]*z, z], -1).reshape(-1,3)
    inv = np.linalg.inv(relative_c2w)
    q = xyz @ inv[:3,:3].T + inv[:3,3]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = q[:,0]/q[:,2]*K[0,0]+K[0,2]
        v = q[:,1]/q[:,2]*K[1,1]+K[1,2]
    ui = np.rint(np.nan_to_num(u,nan=-1e6,posinf=1e6,neginf=-1e6)).astype(np.int64)
    vi = np.rint(np.nan_to_num(v,nan=-1e6,posinf=1e6,neginf=-1e6)).astype(np.int64)
    good = finite.ravel() & (q[:,2] > 0)
    order = np.argsort(-q[:,2]); selected = order[good[order]]
    ui, vi = ui[selected], vi[selected]
    colors = np.ascontiguousarray(rgb.reshape(-1,3)[selected])
    warp = np.tile(author.MAGENTA, (H,W,1))
    if author._HAVE_NUMBA:
        author._splat_kernel(ui,vi,colors,warp,2,H,W)
    else:
        flat = warp.reshape(-1,3)
        for dy in range(-2,3):
            for dx in range(-2,3):
                x,y=ui+dx,vi+dy; keep=(x>=0)&(x<W)&(y>=0)&(y<H)
                flat[y[keep]*W+x[keep]]=colors[keep]
    return warp

def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    inp=OUT/'inputs'; inp.mkdir(exist_ok=True)
    state('checking_source')
    if not (OUT/'source.mp4').exists(): shutil.copy2(SOURCE, OUT/'source.mp4')
    assert sha(SOURCE)==sha(OUT/'source.mp4')
    rgbs=read_video(OUT/'source.mp4')
    rgb_hashes=[hashlib.sha256(f.tobytes()).hexdigest() for f in rgbs]
    assert len(set(rgb_hashes)) > 100, 'Reject a repeated image / near-empty temporal input'
    assert sha(OLD/'reference.jpg')=='e5cf501f443322d61a930e235df5c1ec6e91049514a8170043493104269f73e4'
    mae=float(np.abs(rgbs[0].astype(float)-np.asarray(Image.open(OLD/'reference.jpg'))).mean())
    assert mae<4, 'Source first frame must match the original reference'
    for src,dest in [('intrinsics.npy','intrinsics.npy'),('left_c2w.npy','relative_c2w.npy')]:
        shutil.copy2(OLD/src, inp/dest)
    K=np.load(inp/'intrinsics.npy'); poses=np.load(inp/'relative_c2w.npy')
    assert poses.shape==(N,4,4) and np.allclose(poses[:,:3,:3],np.eye(3))
    assert np.allclose(poses[-1,:3,3],[-3.5,0,20])
    save_json(inp/'frame_mapping.json',dict(source_path=str(SOURCE),source_sha256=sha(SOURCE),source_frame_hashes=rgb_hashes,
        frame_indices=list(range(N)),timestamps_seconds=(np.arange(N)/FPS).tolist(),source_unique_frames=len(set(rgb_hashes)),
        source_first_frame_to_original_jpg_mae=mae,source_dimensions=[W,H],fps=FPS,
        mapping='RGB[i], depth[i], relative_c2w[i], guide[i], target[i]; no repeated source or static frame-0 point cloud'))
    depth_path=inp/'depth_sequence.npz'
    if depth_path.exists():
        data=np.load(depth_path);depths=data['depth'];valids=data['valid'];estimated_K=data['moge_intrinsics']
        assert np.array_equal(data['frame_indices'],np.arange(N)) and str(data['source_sha256'])==sha(SOURCE)
    else:
        state('estimating_depth',frame=0,total=N)
        sys.path[:0]=[str(ROOT/'deps/MoGe'), str(ROOT/'deps/utils3d-moge')]
        from moge.model.v2 import MoGeModel
        moge=MoGeModel.from_pretrained(str(ROOT/'models/moge-2-vitl/model.pt')).to('cuda:0').eval()
        depths=[];valids=[];estimated_K=[];start=time.monotonic()
        fov=float(np.degrees(2*np.arctan(W/(2*K[0,0]))))
        for i,rgb in enumerate(rgbs):
            image=torch.from_numpy(rgb.copy()).permute(2,0,1).float().to('cuda:0')/255
            with torch.inference_mode(): prediction=moge.infer(image,num_tokens=2500,fov_x=fov)
            d=prediction['depth'].float().cpu().numpy();m=prediction['mask'].cpu().numpy().astype(bool)
            m &= np.isfinite(d)&(d>0)
            assert d.shape==m.shape==(H,W) and m.mean()>.25
            depths.append(d);valids.append(m);estimated_K.append(prediction['intrinsics'].float().cpu().numpy())
            if i in [0,3] or (i+1)%12==0 or i==N-1:
                state('estimating_depth',frame=i+1,total=N,elapsed_seconds=time.monotonic()-start)
        depths=np.stack(depths);valids=np.stack(valids);estimated_K=np.stack(estimated_K)
        np.savez_compressed(depth_path,depth=depths,valid=valids,moge_intrinsics=estimated_K,frame_indices=np.arange(N),source_sha256=sha(SOURCE))
        del moge,prediction,image;gc.collect();torch.cuda.empty_cache()
    assert depths.shape==valids.shape==(N,H,W)
    state('warping',frame=0,total=N)
    author=load_author();coverage=[];probe_checks=[]
    with imageio.get_writer(str(OUT/'warp.mp4'),fps=FPS,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p') as writer:
        for i in range(N):
            # P0 fix: both appearance and geometry are indexed INSIDE the frame loop.
            warped=warp_frame(rgbs[i],depths[i],valids[i],K,poses[i],author)
            writer.append_data(warped)
            coverage.append(float(np.any(warped!=[255,0,255],axis=-1).mean()))
            if i in [0,20,40,60]:
                frozen=warp_frame(rgbs[0],depths[0],valids[0],K,poses[i],author)
                changed=float(np.abs(warped.astype(float)-frozen.astype(float)).mean())
                probe_checks.append(dict(frame=i,dynamic_vs_frame0_cloud_mae=changed))
                assert i==0 or changed>1, 'Temporal source unexpectedly equivalent to frame-0 cloud'
            if (i+1)%24==0 or i==N-1:state('warping',frame=i+1,total=N)
    assert len(read_video(OUT/'warp.mp4'))==N
    cfg=dict(frames=N,fps=FPS,width=W,height=H,seed=SEED,prompt='crossview',crossview_strength=.8,
        steps=8,video_shift=12,audio_shift=3,acceleration='existing Larry v4 step600 EMA DMD 8-step LoRA',
        runtime='existing pinned DiffSynth c458cb42; H3 weights/architecture unchanged; not native ComfyUI runtime',
        source_reference_type='video',source_and_guide_same_dimensions_and_indices=True,dynamic_object_removal=False,
        camera='exact previous left numeric matrices, applied as relative target c2w offset in EACH moving source-frame camera coordinates',
        camera_endpoint_translation=[-3.5,0,20],camera_rotation='identity',camera_matrix_sha256=sha(inp/'relative_c2w.npy'),
        camera_units='MoGe estimated metres, not calibrated GT; source ego motion is retained in addition to the relative offset',
        warp='author per-frame reprojection, 99.5 depth percentile, z-sorted painter splat radius2, magenta unknown; full original K retained',
        author_source_sha256=sha(ROOT/'deps/ComfyUI-CrossViewWarp/crossview_warp_node.py'),
        depth_model='existing MoGe2 ViT-L, each of124 RGB frames independently inferred with2500 tokens and fixed source FoV')
    save_json(OUT/'config.json',cfg)
    save_json(OUT/'warp_audit.json',dict(source_coverage=coverage,frame0_cloud_comparison=probe_checks,depth_shape=list(depths.shape),
        depth_unique_frames=len({hashlib.sha256(d.tobytes()).hexdigest() for d in depths}),no_source_frame_broadcast=True))
    state('prepared',source_frames=N,depth_frames=N,warp_frames=N)

def context_for(prompt,guide,source,image_latent=None):
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PackedSequenceBuilder
    from h3_trajectory.geometry import augmented_reference
    L=prompt['prompt_embeds'].shape[-2]; assert L==prompt['text_token_tags'].numel()
    lt,lh,lw=guide.shape[2:]
    assert guide.shape==SHAPE and (source is None or source.shape==SHAPE)
    rows=lt*(lh//2)*(lw//2)
    source_rows=rows if source is not None else 0
    ref_blocks=[dict(kind='video',latent_t=lt,latent_h=lh,latent_w=lw,ref_audio_t=0)] if source is not None else []
    image_rows = 0
    if image_latent is not None:
        assert image_latent.shape[:3] == (1,24,1)
        ih,iw = image_latent.shape[-2:]
        image_rows = (ih//2)*(iw//2)
        ref_blocks.append(dict(kind='image',latent_t=1,latent_h=ih,latent_w=iw,ref_audio_t=0))
    ref_rows = source_rows + image_rows
    builder=MiniMaxH3Unit_PackedSequenceBuilder()
    base=builder._build_packed_ref2va(L,lt,lh,lw,round(N/MODEL_FPS*40),ref_blocks)
    base['token_tags'][base['text_pos']]=prompt['text_token_tags'].reshape(-1)
    ti=torch.arange(L);gi=torch.arange(L,L+rows);ri=torch.arange(L+rows,L+rows+ref_rows)
    ai=torch.arange(L+rows+ref_rows,L+rows+ref_rows+len(base['audio_pos']))
    vi=torch.arange(int(ai[-1])+1,int(ai[-1])+1+rows);used=int(vi[-1])+1;length=((used+63)//64)*64
    xyz=torch.zeros(1,length,3,dtype=torch.float64);tags=torch.full((length,),-1,dtype=torch.long)
    for new,old in [(ti,base['text_pos']),(gi,base['img_pos'][ref_rows:]),(ri,base['img_pos'][:ref_rows]),(ai,base['audio_pos']),(vi,base['img_pos'][ref_rows:])]:
        xyz[:,new]=base['img_position_ids'][:,old];tags[new]=base['token_tags'][old]
    assert torch.equal(xyz[:,gi],xyz[:,vi])
    assert xyz[0,gi,0].unique().numel()==lt
    assert xyz[0,ri[:source_rows],0].unique().numel()==(lt if source is not None else 0)
    # Ordinary source video consumes its full duration; aligned guide does not.
    expected_origin=L+(builder._video_t_span(lt) if source is not None else 0)+int(image_latent is not None)
    assert abs(float(xyz[0,gi[0],0])-expected_origin)<1e-9
    references=[guide] if source is None else [guide,source]
    if image_latent is not None:
        assert xyz[0,ri[source_rows:],0].unique().numel()==1
        references.append(image_latent)
    anchors=[augmented_reference(z,target_latent_t=lt,visual_blocks=len(references),seed=SEED+42,factor=.999).to(torch.bfloat16) for z in references]
    source_slot='|ordinary source VIDEO' if source is not None else ''
    save_json(OUT/'guide_audit.json',dict(layout=f'text|aligned guide{source_slot}'+('|appearance IMAGE' if image_rows else '')+'|target audio|target video|pad',
        source_video_shape=list(source.shape) if source is not None else None,guide_video_shape=list(guide.shape),
        source_tokens=source_rows,appearance_image_tokens=image_rows,guide_tokens=rows,target_tokens=rows,text_tokens=L,
        source_time_origin=float(xyz[0,ri[0],0]) if source_rows else None,target_time_origin=float(xyz[0,vi[0],0]),
        guide_target_all_thw_equal=True,source_and_guide_distinct_time_planes=lt,anchor_dtype=str(anchors[0].dtype),frame_idx=0))
    return dict(imgvid_cond_noise_aug=.999,audio_cond_noise_aug=1.,prompt_embeds=prompt['prompt_embeds'],
        packed=dict(img_pos=torch.cat([gi,ri,vi]),audio_pos=ai,text_pos=ti,img_position_ids=xyz,token_tags=tags,cu_seqlens=torch.tensor([0,used,length],dtype=torch.int32),seq_len=length),
        ref_visual_anchor=torch.cat(anchors),ref_audio_anchor=None,key_log_bias=None)

def attach_crossview(model):
    from safetensors.torch import load_file
    from h3_trajectory.acceleration import _attach_converted,interleave_qkv_rows
    raw=load_file(str(WEIGHTS/'MiniMax-H3_Ref2VA-LoRA-CrossView-Warp_v1_3500.safetensors'),device='cpu')
    names=sorted(k.removesuffix('.lora_down.weight') for k in raw if k.endswith('.lora_down.weight'))
    assert len(names)==200 and len(raw)==600
    converted={}
    for p in names:
        match=re.fullmatch(r'lora_unet_blocks_(\d+)_(attn_qkv_proj|attn_out_proj|mlp_fc1|mlp_fc2)',p);assert match,p
        i,part=match.groups();target='blocks.'+i+'.'+{'attn_qkv_proj':'attn.qkv_proj','attn_out_proj':'attn.out_proj','mlp_fc1':'mlp.fc1','mlp_fc2':'mlp.fc2'}[part]
        a,b=raw[p+'.lora_down.weight'],raw[p+'.lora_up.weight'];assert a.shape[0]==32 and float(raw[p+'.alpha'])==32
        if part=='attn_qkv_proj':b=interleave_qkv_rows(b)
        converted[target+'.lora_A.weight']=a;converted[target+'.lora_B.weight']=b
    assert len(_attach_converted(model,converted,scale=1.,gain=.8,adapter_name='crossview'))==200

def infer(*, warp_only=False, video_shift=12, steps=8, reference_image=None, first_frame_image=None):
    from h3_trajectory.runtime import load_pipe,move_context,velocity
    from h3_trajectory.acceleration import attach_larry
    from h3_trajectory.placement import place_model,synchronize
    from h3_trajectory.sampling import SamplingConfig,sample
    from h3_trajectory.cli import write_video
    from diffsynth.pipelines.minimax_h3_audio_video import MiniMaxH3Unit_PromptEmbedder, MiniMaxH3Unit_ReferenceEncoder
    assert (OUT/'config.json').is_file() and not (OUT/'generated.mp4').exists()
    source_hash=None if warp_only else sha(OUT/'source.mp4')
    warp_hash=sha(OUT/'warp.mp4')
    image_hash=sha(reference_image) if reference_image is not None else None
    first_frame_hash=sha(first_frame_image) if first_frame_image is not None else None
    cache_dir=OUT/'cache';cache_dir.mkdir(exist_ok=True);cache_path=cache_dir/'encoded.pt'
    if cache_path.exists():
        cache=torch.load(cache_path,map_location='cpu',weights_only=True)
        assert cache['source_sha256']==source_hash and cache['warp_sha256']==warp_hash
        assert cache.get('prompt', 'crossview') == PROMPT
        assert cache.get('image_sha256') == image_hash and cache['seed'] == SEED
        assert cache.get('first_frame_sha256') == first_frame_hash
    else:
        state('encoding_vae')
        source_frames=None if warp_only else [Image.fromarray(f) for f in read_video(OUT/'source.mp4')]
        warp_frames=[Image.fromarray(f) for f in read_video(OUT/'warp.mp4')]
        vae=load_pipe('Ref2VA',components=('video_vae',),device='cuda:0',model_root=str(MODEL))
        def encode(frames,image=False):
            pix=vae.preprocess_image(frames,torch_dtype=torch.float32,min_value=0).to('cuda:0') if image else vae.preprocess_video(frames,torch_dtype=torch.float32,min_value=0,device='cuda:0')
            assert image or pix.shape==(1,3,N,H,W)
            with torch.inference_mode():return vae.video_vae.encode_video(pix,dtype=torch.bfloat16,process_image=image,tiled=True,tile_size=256,tile_overlap=64).to('cpu',torch.bfloat16)
        guide=encode(warp_frames)
        source=None if warp_only else encode(source_frames)
        first_image=(warp_frames if warp_only else source_frames)[0]
        if first_frame_image is not None:
            with Image.open(first_frame_image) as im:
                if im.size != (W,H):
                    raise ValueError(f'first-frame image must be {W}x{H}')
                first_image=im.convert('RGB')
        first=encode(first_image,True)
        for reference in (guide,) if warp_only else (guide,source):
            assert tuple(reference.shape)==SHAPE and torch.isfinite(reference).all()
            assert not torch.equal(reference[:,:,0],reference[:,:,20])
        appearance, appearance_latent = None, None
        if reference_image is not None:
            with Image.open(reference_image) as im:
                size = MiniMaxH3Unit_ReferenceEncoder()._resolve_reference_image_shape(vae,*im.size,384)
                appearance = im.convert('RGB').resize(size,Image.Resampling.LANCZOS)
            appearance_latent = encode(appearance,True)
        del vae;gc.collect();torch.cuda.empty_cache()
        state('encoding_prompt_video')
        text=load_pipe('Ref2VA',components=('text_encoder',),device='cuda:0',model_root=str(MODEL))
        latent_t,latent_h,latent_w=guide.shape[2:]
        ref_blocks=[] if warp_only else [dict(kind='video',latent_t=latent_t,latent_h=latent_h,latent_w=latent_w,ref_audio_t=0,prepared_frames=source_frames)]
        if appearance_latent is not None:
            ih,iw = appearance_latent.shape[-2:]
            ref_blocks.append(dict(kind='image',latent_t=1,latent_h=ih,latent_w=iw,ref_audio_t=0,
                                   prepared_image=appearance))
        pdata=MiniMaxH3Unit_PromptEmbedder().process(text,PROMPT,ref_blocks=ref_blocks,height=H,width=W)
        context=context_for(move_context(pdata,'cpu'),guide,source,appearance_latent)
        cache=dict(seed=SEED,warp_latents=guide,visible_mask=torch.zeros((1,1,latent_t,latent_h,latent_w),dtype=torch.bfloat16),
            first_latent=first,audio_shape=(2,32,round(N/MODEL_FPS*40)),context=context,source_sha256=source_hash,warp_sha256=warp_hash,prompt=PROMPT,image_sha256=image_hash,first_frame_sha256=first_frame_hash)
        torch.save(cache,cache_path)
        del text,pdata,context,source,guide,first,source_frames,warp_frames;gc.collect();torch.cuda.empty_cache()
    state('loading_dit')
    pipe=load_pipe('Ref2VA',components=('dit',),device='cpu',model_root=str(MODEL))
    attach_crossview(pipe.dit);attach_larry(pipe.dit,WEIGHTS/'minimax_h3_turbo_v4_step600_ema.safetensors')
    devices=list(DEVICES);place_model(pipe.dit,devices,mlp_chunk_tokens=2048);pipe.device=torch.device('cuda:0')
    settings=SamplingConfig(steps=steps,video_shift=video_shift,audio_shift=3,guide_steps=0,reference_steps=steps,lora_steps=steps)
    state('sampling',step=0,steps=steps);synchronize(devices);start=time.monotonic()
    with torch.inference_mode():
        latent=sample(cache,lambda ctx,v,a,sv,sa:velocity(pipe,ctx,v,a,sv,sa),settings,device='cuda:0',progress=lambda i,n:state('sampling',step=i,steps=n))
    synchronize(devices);elapsed=time.monotonic()-start
    torch.save(latent.cpu(),cache_dir/'generated_latent.pt')
    del latent,cache;pipe.dit.to('cpu');del pipe;gc.collect();torch.cuda.empty_cache()
    state('decoding')
    vae=load_pipe('Ref2VA',components=('video_vae',),device='cuda:0',model_root=str(MODEL))
    latent=torch.load(cache_dir/'generated_latent.pt',map_location='cuda:0',weights_only=True)
    with torch.inference_mode():frames=vae.video_vae.decode_video(latent,dtype=torch.bfloat16,tiled=True,tile_size=256,tile_overlap=64).cpu()
    write_video(frames,OUT/'generated.mp4',N,FPS)
    save_json(OUT/'generation.json',dict(sampling=settings.__dict__,sampler='existing h3_trajectory.sample flow Euler; unchanged from prior8step tests',sampling_seconds=elapsed,generated_sha256=sha(OUT/'generated.mp4'),source_sha256=source_hash,warp_sha256=warp_hash,appearance_image_sha256=image_hash,first_frame_sha256=first_frame_hash,seed=SEED,source_reference_type=None if warp_only else 'video',first_frame_anchor='image' if first_frame_image is not None else ('warp' if warp_only else 'source'),prompt=PROMPT,devices=devices))
    state('generated',sampling_seconds=elapsed)

def compose(inputs=None):
    if inputs is None:
        inputs=[(OUT/'source.mp4','SOURCE VIDEO'),(OUT/'warp.mp4','PER-FRAME WARP'),(OUT/'generated.mp4','H3 CROSSVIEW OUTPUT')]
    streams=[(read_video(path),label) for path,label in inputs]
    with imageio.get_writer(str(OUT/'comparison.mp4'),fps=FPS,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p') as writer:
        for i in range(N):
            canvas=np.full((H+40,W*len(streams),3),24,np.uint8)
            for j,(frames,label) in enumerate(streams):
                canvas[40:,j*W:(j+1)*W]=frames[i]
                cv2.putText(canvas,f'{label} | frame{i:03d} {i/FPS:.3f}s',(j*W+12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
            writer.append_data(canvas)
            if i in [0,20,40,60]:Image.fromarray(canvas).save(OUT/f'comparison_frame_{i:03d}.png')
    state('awaiting_visual_review',videos=[str(path) for path,_ in inputs]+['comparison.mp4'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','infer','compose','run']);args=parser.parse_args()
    try:
        if args.stage=='run':
            infer();compose()
        else:
            globals()[args.stage]()
    except Exception as exc:
        if OUT.is_dir():state('failed',stage=args.stage,error=repr(exc))
        raise
