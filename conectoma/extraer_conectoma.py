"""Extrae del conectoma FlyWire v783 los recuentos de sinapsis usados en la Tabla I (sin umbral mínimo por conexión).
Entradas (descargar antes):
  - Connectivity_783.parquet: https://github.com/philshiu/Drosophila_brain_model
  - Supplemental_file1_neuron_annotations.tsv: https://github.com/flyconnectome/flywire_annotations (supplemental_files/)
Uso: python conectoma/extraer_conectoma.py RUTA/Supplemental_file1_neuron_annotations.tsv RUTA/Connectivity_783.parquet"""
import sys
import pandas as pd, numpy as np, json, re
a=pd.read_csv(sys.argv[1],sep='\t',low_memory=False,usecols=['root_id','cell_type','side','top_nt','super_class'])
c=pd.read_parquet(sys.argv[2])
print(c.columns.tolist(), len(c))
pre,post,w=[x for x in c.columns if 'Pre' in x and 'ID' in x][0],[x for x in c.columns if 'Post' in x and 'ID' in x][0],'Connectivity'
t=dict(zip(a.root_id,a.cell_type))
c['pre_t']=c[pre].map(t); c['post_t']=c[post].map(t)
def ins(target):
    s=c[c.post_t==target].groupby('pre_t')[w].sum().sort_values(ascending=False); return s
print("tipos LPi:", sorted({x for x in a.cell_type.dropna() if str(x).startswith('LPi')})[:20])
lp=ins('LPLC2'); gf=ins('DNp01'); lc4=ins('LC4')
print("Entradas a LPLC2 (top 20):\n", lp.head(20).to_string())
print("Entradas a DNp01 (top 15):\n", gf.head(15).to_string())
print("Entradas a LC4 (top 10):\n", lc4.head(10).to_string())
nt=dict(zip(a.cell_type,a.top_nt))
out=dict(
 n_LPLC2=int((a.cell_type=='LPLC2').sum()), n_LC4=int((a.cell_type=='LC4').sum()),
 T4_LPLC2={k:int(lp.get(k,0)) for k in ['T4a','T4b','T4c','T4d']},
 T5_LPLC2={k:int(lp.get(k,0)) for k in ['T5a','T5b','T5c','T5d']},
 LPi_LPLC2={k:int(v) for k,v in lp.items() if str(k).startswith('LPi')},
 LPLC2_total_in=int(lp.sum()),
 GF_in={k:int(v) for k,v in gf.head(15).items()},
 GF_total_in=int(gf.sum()),
 nt={k:nt.get(k) for k in ['T4a','T5a','LPLC2','LC4','DNp01']+[k for k in lp.index if str(k).startswith('LPi')]},
)
json.dump(out,open('conectoma_resumen.json','w'),indent=1,ensure_ascii=False)
print(json.dumps(out,indent=1,ensure_ascii=False))
