"""
Validation, signature, confounder and comparator analyses (GSE75010) for:
"Cross-cohort replication of hypoxia-related placental gene expression changes in preeclampsia".
Inputs (same folder or edit paths below):
  GSE75010_series_matrix_txt.gz   (GEO series matrix)
  GPL6244-17930.txt               (GEO platform annotation table)
  discovery_vs_validation_limma.csv  (output of GSE75010_limma_validation.R: columns gene, lfc_d, padj_d, lfc_v, fdr_v, replicated)
Run: python validation_confounder_analysis.py
"""
import gzip, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
SM='GSE75010_series_matrix_txt.gz'; ANN='GPL6244-17930.txt'; DV='discovery_vs_validation_limma.csv'

# ---- 1. parse GEO series matrix and map probes to genes
lines=gzip.open(SM,'rt').read().split('\n')
b=[i for i,l in enumerate(lines) if l.startswith('!series_matrix_table_begin')][0]
e=[i for i,l in enumerate(lines) if l.startswith('!series_matrix_table_end')][0]
meta={}
for l in lines[:b]:
    v=[x.strip('"') for x in l.split('\t')[1:]]
    if l.startswith('!Sample_geo_accession'): gsm=v
    elif l.startswith('!Sample_title'): meta['title']=v
    elif l.startswith('!Sample_characteristics_ch1'):
        meta[v[0].split(':')[0].strip()]=[t.split(':',1)[1].strip() for t in v]
m=pd.DataFrame(meta,index=gsm)
x=pd.read_csv(SM,sep='\t',skiprows=b+1,nrows=e-b-2,index_col=0,compression='gzip'); x.columns=[c.strip('"') for c in x.columns]; x.index=[str(i).strip('"') for i in x.index]
ann=pd.read_csv(ANN,sep='\t',comment='#',dtype=str,usecols=['ID','gene_assignment','category'])
def sym(s):
    if not isinstance(s,str) or s.startswith('---'): return None
    f=s.split(' /// ')[0].split(' // '); return f[1].strip() if len(f)>1 and f[1].strip() not in ('---','') else None
ann['gene']=ann.gene_assignment.map(sym); ann=ann[ann.gene.notna()&(ann.category.str.strip()=='main')].set_index(ann.ID.str.strip()[ann.gene.notna()&(ann.category.str.strip()=='main')])
x=x.loc[x.index.intersection(ann.index)]; x['gene']=ann.loc[x.index,'gene']; x['mu']=x.drop(columns='gene').mean(axis=1)
x=x.sort_values('mu',ascending=False).drop_duplicates('gene').set_index('gene').drop(columns='mu')[m.index]

# ---- 2. covariates
y=(m['diagnosis']=='PE').astype(int).values
ga=pd.to_numeric(m['ga (week)']).values+pd.to_numeric(m['ga (day)']).values/7
sex=(m['infant gender']=='M').astype(float).values
cs=(m['mode of delivery']=='C-Section').astype(float).values; av=(m['attempted vaginal delivery']=='Yes').astype(float).values
ch=(m['chorioamnionitis diagnosis']=='Yes').astype(float).values
chyp=m['title'].str.contains('-CH').values          # CH = chronic hypertension (Leavey et al. 2016, Fig. 1C)
sga=pd.to_numeric(m['newborn weight z-score']).values<-1.28

# ---- 3. signature (genes chosen from the DISCOVERY data only)
d=pd.read_csv(DV); sel=d[(d.padj_d<0.05)&(d.lfc_d.abs()>0.585)]
z=x.loc[sel.gene].apply(lambda r:(r-r.mean())/r.std(),axis=1)
score=(z.loc[sel.gene[sel.lfc_d>0]].mean()-z.loc[sel.gene[sel.lfc_d<0]].mean()).values
print('signature genes on array:',len(sel),'up',(sel.lfc_d>0).sum(),'down',(sel.lfc_d<0).sum())

def auc(s,yy): r=stats.rankdata(s); n1=yy.sum(); return (r[yy==1].sum()-n1*(n1+1)/2)/(n1*(len(yy)-n1))
def resid(s,covs,k=None):
    k=np.ones(len(y),bool) if k is None else k
    Z=np.column_stack([np.ones(k.sum())]+[c[k] for c in covs]); return s[k]-Z@np.linalg.lstsq(Z,s[k],rcond=None)[0]
def boot_ci(s,covs,n=1000,seed=1):
    r=resid(s,covs); rng=np.random.default_rng(seed)
    return np.percentile([auc(r[i],y[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(n))],[2.5,97.5])
print('\n== Table 2 ==')
for lab,covs in [('GA+sex (primary)',[ga,sex]),('+ mode',[ga,sex,cs]),('+ mode + labour',[ga,sex,cs,av]),('+ mode + labour + chorioamnionitis',[ga,sex,cs,av,ch])]:
    print('%-40s AUC %.3f (95%% CI %.3f-%.3f)'%(lab,auc(resid(score,covs),y),*boot_ci(score,covs)))
clean=(cs==1)&(av==0)&(ch==0)
print('restricted subgroup n=%d: raw %.3f, adjusted %.3f'%(clean.sum(),auc(score[clean],y[clean]),auc(resid(score,[ga,sex],clean),y[clean])))
for lab,k in [('preterm <37 wk',ga<37),('excluding SGA (z<-1.28)',~sga),('excluding chronic hypertension',~chyp),('chronic hypertension only',chyp)]:
    print('%-32s n=%3d raw %.3f adjusted %.3f'%(lab,k.sum(),auc(score[k],y[k]),auc(resid(score,[ga,sex],k),y[k])))

# ---- 4. comparators (5-fold CV logistic regression, 20 repeats) and paired bootstrap vs single genes
def cvauc(X):
    X=(X-X.mean(0))/X.std(0); o=[]
    for sd in range(20):
        p=cross_val_predict(LogisticRegression(max_iter=1000),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=sd),method='predict_proba')[:,1]; o.append(roc_auc_score(y,p))
    return np.mean(o)
print('\nCV AUC: GA+sex %.3f | signature %.3f | signature+GA+sex %.3f'%(cvauc(np.column_stack([ga,sex])),cvauc(score[:,None]),cvauc(np.column_stack([score,ga,sex]))))
for g in ['FLT1','FSTL3']:
    gv=x.loc[g].values
    for lab,(a,bb) in [('raw',(score,gv)),('adjusted',(resid(score,[ga,sex]),resid(gv,[ga,sex])))]:
        rng=np.random.default_rng(1); diff=[auc(a[i],y[i])-auc(bb[i],y[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(2000))]
        print('signature minus %s (%s): AUC %.3f vs %.3f, difference %.3f (95%% CI %.3f to %.3f)'%(g,lab,auc(a,y),auc(bb,y),auc(a,y)-auc(bb,y),*np.percentile(diff,[2.5,97.5])))
def ll(X):
    p=LogisticRegression(C=1e6,max_iter=3000).fit(X,y).predict_proba(X)[:,1]; return -(y*np.log(p)+(1-y)*np.log(1-p)).sum()
fl=x.loc['FLT1'].values; l_b=ll(np.column_stack([ga,sex])); l_s=ll(np.column_stack([score,ga,sex])); l_f=ll(np.column_stack([fl,ga,sex])); l_sf=ll(np.column_stack([score,fl,ga,sex]))
print('LR tests: signature over GA+sex chi2=%.1f p=%.1e | FLT1 added to signature+GA+sex p=%.2f | signature added to FLT1+GA+sex p=%.2f'%(2*(l_b-l_s),stats.chi2.sf(2*(l_b-l_s),1),stats.chi2.sf(2*(l_s-l_sf),1),stats.chi2.sf(2*(l_f-l_sf),1)))

# ---- 5. random gene sets and permutation test of direction agreement
rng=np.random.default_rng(0); nn=[]
for _ in range(300):
    gs=rng.choice(x.index.values,len(sel),replace=False); zz=x.loc[gs].apply(lambda r:(r-r.mean())/r.std(),axis=1); k=int(.83*len(gs))
    a=roc_auc_score(y,zz.iloc[:k].mean().values-zz.iloc[k:].mean().values); nn.append(max(a,1-a))
print('\nrandom 83-gene sets: 95th percentile %.3f, max %.3f'%(np.percentile(nn,95),max(nn)))
disc=d[d.padj_d<0.05]; X=x.loc[disc.gene].values; obs=(np.sign(disc.lfc_d.values)==np.sign(disc.lfc_v.values)).mean(); null=[]
for _ in range(500):
    yy=rng.permutation(y); dd=X[:,yy==1].mean(1)-X[:,yy==0].mean(1); null.append((np.sign(disc.lfc_d.values)==np.sign(dd)).mean())
print('direction agreement %.3f; permutation null mean %.2f (95%% %.2f-%.2f, max %.2f); P=%.4f'%(obs,np.mean(null),*np.percentile(null,[2.5,97.5]),max(null),(1+sum(v>=obs for v in null))/501))

# ---- 6. mean signature score by group (Supplementary Table 5)
t=pd.DataFrame({'dx':np.where(y==1,'PE','non-PE'),'CH':np.where(chyp,'chronic hypertension','no chronic hypertension'),'score':score,
                'delivery':np.where(cs==1,'caesarean','vaginal'),'chorioamnionitis':np.where(ch==1,'yes','no')})
print('\nmean score by diagnosis x chronic hypertension'); print(t.groupby(['dx','CH']).score.agg(['mean','std','count']).round(2))
print('\nby diagnosis x delivery mode'); print(t.groupby(['dx','delivery']).score.agg(['mean','count']).round(2))
print('\nnon-PE by chorioamnionitis'); print(t[t.dx=='non-PE'].groupby('chorioamnionitis').score.agg(['mean','count']).round(2))
t.to_csv('signature_scores_by_sample.csv',index=False)

# ================= 7. NEW ANALYSES (revision) =================
from scipy.stats import norm
def delong(y,s1,s2):
    pos=y==1; m=pos.sum(); n=(~pos).sum()
    def comp(s):
        a=s[pos]; b=s[~pos]
        psi=(a[:,None]>b[None,:]).astype(float)+0.5*(a[:,None]==b[None,:])
        return psi.mean(), psi.mean(1), psi.mean(0)
    A1,v10_1,v01_1=comp(s1); A2,v10_2,v01_2=comp(s2)
    S10=np.cov(np.vstack([v10_1,v10_2])); S01=np.cov(np.vstack([v01_1,v01_2]))
    S=S10/m+S01/n; dd=A1-A2; var=S[0,0]+S[1,1]-2*S[0,1]
    zz=dd/np.sqrt(var); return A1,A2,dd,zz,2*norm.sf(abs(zz))
print('\n== DeLong tests (GA+sex-residualised scores) ==')
rs=resid(score,[ga,sex])
for g in ['FLT1','FSTL3']:
    rg=resid(x.loc[g].values,[ga,sex]); A1,A2,dd,zz,p=delong(y,rs,rg)
    print('signature %.3f vs %s %.3f: diff %.3f, z=%.2f, DeLong P=%.3f'%(A1,g,A2,dd,zz,p))
# AUC drop primary -> +chorioamnionitis adjusted (paired bootstrap)
r1=resid(score,[ga,sex]); r2=resid(score,[ga,sex,cs,av,ch]); rng=np.random.default_rng(3)
dr=[auc(r1[i],y[i])-auc(r2[i],y[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(2000))]
print('AUC drop GA+sex -> +mode+labour+chorio: %.3f (95%% CI %.3f to %.3f)'%(auc(r1,y)-auc(r2,y),*np.percentile(dr,[2.5,97.5])))

# fully adjusted per-gene model (OLS, equivalent to limma without moderation) for all discovery FDR<0.05 genes on array
def ols_group(Y,design_cov,yv):
    Xf=np.column_stack([np.ones(len(yv)),yv]+design_cov); beta,_,_,_=np.linalg.lstsq(Xf,Y.T,rcond=None)
    res=Y.T-Xf@beta; df=len(yv)-Xf.shape[1]; s2=(res**2).sum(0)/df; XtXi=np.linalg.inv(Xf.T@Xf)
    se=np.sqrt(s2*XtXi[1,1]); t=beta[1]/se; return beta[1],2*stats.t.sf(abs(t),df)
disc=d[d.padj_d<0.05].copy(); Xd=x.loc[disc.gene].values
covfull=[ga,sex,cs,av,ch]
b,p=ols_group(Xd,covfull,y.astype(float))
def bh(p):
    o=np.argsort(p); r=np.empty(len(p)); q=p[o]*len(p)/np.arange(1,len(p)+1); q=np.minimum.accumulate(q[::-1])[::-1]; r[o]=np.minimum(q,1); return r
fdr=bh(p); same=np.sign(b)==np.sign(disc.lfc_d.values)
print('\n== Fully adjusted model (GA, sex, delivery mode, attempted vaginal delivery, chorioamnionitis) ==')
print('genes tested %d; same direction %d (%.1f%%); nominal P<0.05 & same direction %d; FDR<0.05 & same direction %d'%(len(disc),same.sum(),100*same.mean(),((p<0.05)&same).sum(),((fdr<0.05)&same).sum()))
pd.DataFrame({'gene':disc.gene.values,'discovery_log2FC':disc.lfc_d.values,'adj_logFC_GSE75010':b,'P':p,'FDR':fdr,'same_direction':same}).sort_values('P').to_csv('fully_adjusted_model_364genes.csv',index=False)

# Freedman-Lane permutation: adjusted model refitted in every permutation
def fl_perm(covs,nperm=1000,seed=7):
    Z=np.column_stack([np.ones(len(y))]+covs); Yg=Xd.T  # samples x genes
    H=Z@np.linalg.pinv(Z); fit=H@Yg; res=Yg-fit
    Xf=np.column_stack([Z,y.astype(float)]); P_=np.linalg.pinv(Xf)
    obs=(np.sign((P_@Yg)[-1])==np.sign(disc.lfc_d.values)).mean(); rng=np.random.default_rng(seed); nl=[]
    for _ in range(nperm):
        Yp=fit+res[rng.permutation(len(y))]; nl.append((np.sign((P_@Yp)[-1])==np.sign(disc.lfc_d.values)).mean())
    nl=np.array(nl); return obs,nl.mean(),np.percentile(nl,[2.5,97.5]),nl.max(),(1+(nl>=obs).sum())/(nperm+1)
print('\n== Freedman-Lane permutation (direction agreement, 1000 perms) ==')
for lab,covs in [('GA+sex',[ga,sex]),('fully adjusted',covfull)]:
    o,mn,ci,mx,pv=fl_perm(covs); print('%s: observed %.3f; null mean %.3f (95%% %.3f-%.3f; max %.3f); P=%.4f'%(lab,o,mn,*ci,mx,pv))

# ================= 8. NON-LINEAR GESTATIONAL-AGE SENSITIVITY =================
def _ns(g,knots): return [g,g**2,g**3]+[np.clip(g-k,0,None)**3 for k in knots]
_kn=np.percentile(ga,[25,50,75]); _c=(ga-ga.mean())/ga.std()
print('\n== Non-linear gestational-age adjustment ==')
for lab,covs in [('linear',[ga,sex]),('quadratic',[ga,_c**2,sex]),('cubic',[ga,_c**2,_c**3,sex]),('cubic spline',_ns(ga,_kn)+[sex])]:
    rs=resid(score,covs); rf=resid(x.loc['FLT1'].values,covs); rng=np.random.default_rng(1)
    ci=np.percentile([auc(rs[i],y[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(1000))],[2.5,97.5])
    print('%-13s signature AUC %.3f (%.3f-%.3f); FLT1 %.3f'%(lab,auc(rs,y),*ci,auc(rf,y)))
Xf=np.column_stack([np.ones(len(y)),y]+_ns(ga,_kn)+[sex]); bb=np.linalg.lstsq(Xf,Xd.T,rcond=None)[0][1]
print('direction agreement with spline GA: %d/%d'%((np.sign(bb)==np.sign(disc.lfc_d.values)).sum(),len(disc)))
