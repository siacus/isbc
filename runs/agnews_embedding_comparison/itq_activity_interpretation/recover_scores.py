from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from information_retention_extended import _chunked_projection
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/agnews_qwen_all_fixed_balance'
ids=pd.read_csv(ROOT/'runs/agnews_embedding_comparison/all_fixed_balance_label_name/predictive_split.csv',dtype={'id':str}).id.to_numpy()
raw_ids=pd.read_csv(RUN/'raw_embeddings_agnews_all_ready_best_ids.csv',dtype={'id':str}).id.to_numpy()
ix=pd.Index(raw_ids).get_indexer(ids);assert (ix>=0).all()
raw=np.load(RUN/'raw_embeddings_agnews_all_ready_best.npy')[ix]
from reproduction_paths import qwen_reference_cache
cached=np.load(qwen_reference_cache(ROOT))['itq']
with threadpool_limits(limits=4):
 pca=PCA(n_components=16,svd_solver='randomized',random_state=1234,n_oversamples=32,iterated_power=7).fit(raw)
 scores=_chunked_projection(raw,pca.mean_,pca.components_).astype(float)
 rotation,_=np.linalg.qr(np.random.default_rng(1251).standard_normal((16,16)))
 for it in range(50):
  binary=np.where(scores@rotation>=0,1.,-1.)
  left,_,right=np.linalg.svd(scores.T@binary,full_matrices=False)
  rotation=left@right
 rotated=scores@rotation
 reproduced=(rotated>=0).astype(np.uint8)
 agreement=dict(bit_agreement=float((reproduced==cached).mean()),identical_documents=int((reproduced==cached).all(1).sum()),n_documents=len(ids),max_rotation_orthogonality_error=float(np.abs(rotation.T@rotation-np.eye(16)).max()))
 print(agreement,flush=True)
 np.savez_compressed(OUT/'itq_scores.npz',ids=ids.astype(str),scores=rotated,rotation=rotation,pca_mean=pca.mean_,pca_components=pca.components_,native_codes=reproduced,cached_codes=cached)
 (OUT/'recovery.json').write_text(json.dumps(agreement,indent=2))
