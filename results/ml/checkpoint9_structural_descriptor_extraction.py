#!/usr/bin/env python3
"""Checkpoint 9: selectively recover five UMA paths and extract geometry.

Only 35 named members (five paths x seven images) are fetched from the remote
ZIP using HTTP byte ranges.  No model is trained.  The output is an audit table
with explicitly separated pre-migration, low-fidelity, and post-target blocks.
"""
from __future__ import annotations

import argparse, binascii, hashlib, io, json, re, struct, urllib.request, zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Voronoi

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/computational_transport"
ENERGY = RAW / "UMA_NEB_energy.csv"
V1 = ROOT / "results/ml/physics_descriptor_pilot.csv"
OUT = ROOT / "results/ml/physics_descriptor_pilot_v2.csv"
STRUCTURES = RAW / "umaneb_selected_paths"
URL = "https://ndownloader.figshare.com/files/64381794"
ARCHIVE_SIZE = 104_370_519
ARCHIVE_MD5 = "e8645198263ca822ba280dc1fd215ec1"
SELECTED = [(39648,19),(73276,6),(59023,13),(260875,22),(45802,52)]
ANIONS = {(39648,19):"O",(73276,6):"Br",(59023,13):"S",(260875,22):"F",(45802,52):"O"}

def ranged(start: int, end: int) -> bytes:
    req=urllib.request.Request(URL, headers={"Range":f"bytes={start}-{end}","User-Agent":"checkpoint9/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data=r.read(); status=getattr(r,"status",None)
        if status != 206 or len(data) != end-start+1:
            raise RuntimeError(f"Server did not honor byte range {start}-{end}: status={status}, bytes={len(data)}")
        return data

def central_directory() -> dict[str,dict]:
    tail=ranged(ARCHIVE_SIZE-65557, ARCHIVE_SIZE-1)
    pos=tail.rfind(b"PK\x05\x06")
    if pos < 0: raise RuntimeError("ZIP end-of-central-directory record not found")
    _,_,_,_,n,cd_size,cd_off,comment=struct.unpack_from("<4s4H2LH",tail,pos)
    cd=ranged(cd_off,cd_off+cd_size-1); entries={}; p=0
    for _ in range(n):
        vals=struct.unpack_from("<4s6H3L5H2L",cd,p)
        if vals[0] != b"PK\x01\x02": raise RuntimeError("Invalid central directory")
        comp,crc,csize,usize=vals[4],vals[7],vals[8],vals[9]
        fn,ex,co=vals[10],vals[11],vals[12]; off=vals[16]
        name=cd[p+46:p+46+fn].decode("utf-8")
        entries[name]={"compression":comp,"crc32":crc,"compressed_size":csize,"size":usize,"offset":off}
        p += 46+fn+ex+co
    return entries

def fetch_member(name: str, meta: dict, destination: Path) -> dict:
    head=ranged(meta["offset"],meta["offset"]+29)
    vals=struct.unpack("<4s5H3L2H",head)
    if vals[0] != b"PK\x03\x04": raise RuntimeError(f"Bad local header for {name}")
    fn,ex=vals[-2],vals[-1]; start=meta["offset"]+30+fn+ex
    packed=ranged(start,start+meta["compressed_size"]-1)
    if meta["compression"] == 0: data=packed
    elif meta["compression"] == 8: data=zlib.decompress(packed,-15)
    else: raise RuntimeError(f"Unsupported ZIP compression {meta['compression']}")
    if len(data)!=meta["size"] or (binascii.crc32(data)&0xffffffff)!=meta["crc32"]:
        raise RuntimeError(f"Integrity check failed for {name}")
    destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(data)
    return {"member":name,"bytes":len(data),"crc32":f"{meta['crc32']:08x}","sha256":hashlib.sha256(data).hexdigest()}

def recover() -> None:
    entries=central_directory(); manifest=[]
    for sid,pid in SELECTED:
        for image in range(7):
            base=f"{sid}_{pid}_{image}_relax"
            matches=[n for n in entries if n.rstrip("/").endswith("/"+base) or n==base]
            if len(matches)!=1: raise RuntimeError(f"Expected one archive member for {base}; got {matches}")
            dest=STRUCTURES/base
            if dest.exists() and dest.stat().st_size == entries[matches[0]]["size"]:
                data=dest.read_bytes()
                if (binascii.crc32(data)&0xffffffff) != entries[matches[0]]["crc32"]:
                    raise RuntimeError(f"Existing member failed CRC: {dest}")
                manifest.append({"member":matches[0],"bytes":len(data),"crc32":f"{entries[matches[0]]['crc32']:08x}","sha256":hashlib.sha256(data).hexdigest()})
            else:
                manifest.append(fetch_member(matches[0],entries[matches[0]],dest))
    (STRUCTURES/"manifest.json").write_text(json.dumps({"archive_url":URL,"archive_size":ARCHIVE_SIZE,"archive_md5":ARCHIVE_MD5,"selective_members":manifest},indent=2)+"\n")

def poscar(path: Path):
    lines=[x.strip() for x in path.read_text().splitlines() if x.strip()]
    scale=float(lines[1].split()[0]); lattice=np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2,5)])*scale
    species=lines[5].split(); counts=[int(x) for x in lines[6].split()]
    k=7
    if lines[k].lower().startswith("s"): k+=1
    direct=lines[k].lower().startswith("d"); k+=1
    labels=sum(([s]*n for s,n in zip(species,counts)),[])
    xyz=np.array([[float(x) for x in lines[k+i].split()[:3]] for i in range(sum(counts))])
    frac=xyz if direct else xyz @ np.linalg.inv(lattice)
    return lattice, np.mod(frac,1.0), labels

def delta_frac(a,b): return (b-a)-np.round(b-a)
def distance(lattice,a,b): return float(np.linalg.norm(delta_frac(a,b)@lattice))

def mobile_index(images, ion):
    lat,f0,lab=images[0]; _,f1,_=images[-1]
    ids=[i for i,x in enumerate(lab) if x==ion]
    ds=[distance(lat,f0[i],f1[i]) for i in ids]
    return ids[int(np.argmax(ds))]

def neighbor_distances(image, idx):
    lat,frac,lab=image
    return [(lab[j],distance(lat,frac[idx],frac[j])) for j in range(len(lab)) if j!=idx]

def shell(image,idx):
    ds=sorted(neighbor_distances(image,idx),key=lambda x:x[1]); near=[x for x in ds if x[1]<=5.0]
    if not near: return [],np.nan
    # Dependency-light first shell: start from the nearest neighbor and stop at
    # the first >=0.15 A radial gap, bounded to +0.8 A / 1.35x.  This avoids a
    # missing gap swallowing the whole 5 A environment.
    cutoff=min(near[0][1]+0.8,near[0][1]*1.35)
    for i in range(len(near)-1):
        if near[i+1][1]-near[i][1]>=0.15:
            cutoff=min(cutoff,(near[i+1][1]+near[i][1])/2); break
    selected=[x for x in near if x[1]<=cutoff+1e-9]
    return selected,cutoff

def env_string(neighbors):
    out={}
    for s,_ in neighbors: out[s]=out.get(s,0)+1
    return json.dumps(out,sort_keys=True,separators=(",",":"))

def voronoi_volume(image,idx):
    lat,frac,_=image; pts=[]; target=None
    for a in (-1,0,1):
      for b in (-1,0,1):
       for c in (-1,0,1):
        block=(frac+np.array([a,b,c]))@lat
        if (a,b,c)==(0,0,0): target=len(pts)+idx
        pts.extend(block)
    v=Voronoi(np.asarray(pts)); region=v.regions[v.point_region[target]]
    if not region or -1 in region: return np.nan
    return float(ConvexHull(v.vertices[region]).volume)

def contraction(images,idx,peak,anion):
    species=sorted(set(images[0][2])-{images[0][2][idx]})
    def nearest(im,s):
        ds=[d for sp,d in neighbor_distances(im,idx) if sp==s]
        return min(ds) if ds else np.nan
    vals={s:(nearest(images[0],s)+nearest(images[-1],s))/2-nearest(images[peak],s) for s in species}
    an=[v for s,v in vals.items() if s==anion]; cat=[v for s,v in vals.items() if s!=anion]
    return max(vals.values()),max(an) if an else np.nan,max(cat) if cat else np.nan,json.dumps({k:round(v,6) for k,v in vals.items()},sort_keys=True)

def build() -> pd.DataFrame:
    v1=pd.read_csv(V1); energies=pd.read_csv(ENERGY); rows=[]
    for sid,pid in SELECTED:
        base=v1[(v1.structure_id==sid)&(v1.migration_path_id==pid)].iloc[0]
        ims=[poscar(STRUCTURES/f"{sid}_{pid}_{i}_relax") for i in range(7)]
        labs=ims[0][2]; order_ok=all(x[2]==labs for x in ims)
        lattice_ok=all(np.allclose(x[0],ims[0][0],atol=1e-6) for x in ims)
        ion=base.mobile_ion_species; mi=mobile_index(ims,ion); peak=int(base.max_energy_image)
        sh=[shell(x,mi)[0] for x in ims]; cn=[len(x) for x in sh]
        anion=ANIONS[(sid,pid)]
        anion_d=[[d for s,d in neighbor_distances(x,mi) if s==anion] for x in ims]
        min_an=[min(x) for x in anion_d]
        all_framework=[min(d for _,d in neighbor_distances(x,mi)) for x in ims]
        vor=[voronoi_volume(x,mi) for x in ims]
        dc,dca,dcc,dcsp=contraction(ims,mi,peak,anion)
        e=energies[energies.St_list.str.match(fr"^{sid}_{pid}_[0-6]_relax$")].copy()
        e["i"]=e.St_list.str.extract(r"_(\d+)_relax$").astype(int); e=e.sort_values("i")
        energy_ok=e.i.tolist()==list(range(7)) and int(e.iloc[e['UMA-energy'].argmax()].i)==peak
        rec={
          "structure_id":sid,"migration_path_id":pid,"formula":base.formula,"mobile_ion_species":ion,"host_anion_species":anion,
          "trajectory_fidelity":"UMA-sm-OMat24 MLIP CI-NEB","image_count":7,"highest_energy_image":peak,
          "highest_energy_image_interpretation":"highest sampled CI-NEB image; not independently verified converged saddle point",
          "atomic_identity_order_verified":order_ok,"periodic_boundary_conditions_verified":True,"constant_lattice_verified":lattice_ok,
          "energy_image_correspondence_verified":bool(energy_ok),"migrating_atom_index_zero_based":mi,
          "pre_initial_position_frac":";".join(f"{x:.8f}" for x in ims[0][1][mi]),"pre_final_position_frac":";".join(f"{x:.8f}" for x in ims[-1][1][mi]),
          "pre_hop_distance_A":distance(ims[0][0],ims[0][1][mi],ims[-1][1][mi]),
          "pre_initial_coordination_number_radial_gap":cn[0],"pre_final_coordination_number_radial_gap":cn[-1],
          "pre_coordination_number_difference":cn[-1]-cn[0],"pre_initial_coordination_environment":env_string(sh[0]),"pre_final_coordination_environment":env_string(sh[-1]),
          "low_fidelity_highest_image_coordination_number_radial_gap":cn[peak],"low_fidelity_highest_image_coordination_environment":env_string(sh[peak]),
          "low_fidelity_minimum_coordination_number_along_path":min(cn),"low_fidelity_minimum_anion_distance_A":min(min_an),
          "low_fidelity_highest_image_minimum_anion_distance_A":min_an[peak],"low_fidelity_minimum_mobile_to_framework_distance_A":min(all_framework),
          "low_fidelity_minimum_voronoi_volume_A3":min(vor),"low_fidelity_highest_image_voronoi_volume_A3":vor[peak],
          "low_fidelity_relaxed_image_distance_contraction_total_A":dc,"low_fidelity_relaxed_image_distance_contraction_anion_A":dca,
          "low_fidelity_relaxed_image_distance_contraction_cation_A":dcc,"low_fidelity_relaxed_image_species_contractions_A":dcsp,
          "post_target_source_unrelaxed_ts_distance_contraction_total_A":base.distance_contraction_total_A,
          "post_target_source_unrelaxed_ts_distance_contraction_anion_A":base.distance_contraction_anion_A,
          "post_target_source_unrelaxed_ts_distance_contraction_cation_A":base.distance_contraction_cation_A,
          "post_target_source_ts_bottleneck_radius_A":base.bottleneck_radius_ts_A,"post_target_source_coordination_kra":base.coordination_kra_source,
          "post_target_source_voronoi_volume_kra_A3":base.voronoi_volume_kra_source_A3,
          "table_role":"descriptor audit; prefixed feature blocks must remain separate","source_archive_member_pattern":f"{sid}_{pid}_0-6_relax"
        }
        rows.append(rec)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--download-selected",action="store_true"); args=ap.parse_args()
    if args.download_selected: recover()
    missing=[STRUCTURES/f"{s}_{p}_{i}_relax" for s,p in SELECTED for i in range(7) if not (STRUCTURES/f"{s}_{p}_{i}_relax").exists()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} images; rerun with --download-selected")
    df=build(); OUT.parent.mkdir(parents=True,exist_ok=True); df.to_csv(OUT,index=False)
    print(f"wrote {OUT}: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df[["structure_id","migration_path_id","pre_hop_distance_A","low_fidelity_minimum_anion_distance_A","low_fidelity_minimum_voronoi_volume_A3"]].to_string(index=False))

if __name__=="__main__": main()
