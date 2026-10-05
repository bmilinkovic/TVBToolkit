"""Audit legacy AAL90 receptor indexing and produce separate, corrected figures.

Never overwrites production inputs. Raw PET parcels exclude background already;
the first 90 entries (not entries 1:91) correspond to AAL90. Combine repeated
studies by sample-size-weighted regional z scores; min-max each final map for
display/model weights. These weights are relative, not absolute PET densities.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn import plotting


def minmax(x):
    """Scale each column, refusing constant or nonfinite input."""
    x = np.asarray(x, dtype=float)
    span = np.ptp(x, axis=0)
    if not np.isfinite(x).all() or np.any(span <= 0):
        raise ValueError('Nonfinite or constant receptor map')
    return (x - x.min(axis=0)) / span


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/Users/borjan/code/Brain-Act/brain-act'))
    parser.add_argument('--atlas', type=Path, default=Path('/Users/borjan/nilearn_data/aal_SPM12/aal/atlas/AAL.nii'))
    parser.add_argument('--output', type=Path, default=Path('results/receptor_maps_aal90_audit'))
    parser.add_argument('--comparison-csv', type=Path, help='Optional archived table for historical comparisons')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    oldpath = args.comparison_csv or root / 'data/receptors/hansen_receptors_aal90.csv'
    old = pd.read_csv(oldpath, index_col=0)
    elements = ET.parse(args.atlas.with_suffix('.xml')).findall('.//data/label')
    names = [e.findtext('name') for e in elements]
    ids = [int(e.findtext('index')) for e in elements]
    assert names[:90] == list(old.index)
    parcels = args.source / 'results/hansen_receptors/aal90'
    raw, checks = {}, {}
    for col in old.columns:
        candidates = [parcels / (col + suffix) for suffix in ('.csv', '.nii.csv')]
        path = next(p for p in candidates if p.exists())
        values = np.loadtxt(path, delimiter=',')
        assert values.shape == (116,)
        raw[col] = values[:90]
        checks[col] = dict(source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          old_shifted_maxscale_error=float(np.max(abs(values[1:91]/values[1:91].max()-old[col]))))
    raw = pd.DataFrame(raw, index=names[:90])
    # Verify against the actual PET volume, not just file names or assumptions.
    key = '5HT2a_cimbi_hc29_beliveau'
    pet = args.source / ('external/hansen_receptors/data/PET_nifti_images/' + key + '.nii')
    masker = NiftiLabelsMasker(labels_img=str(args.atlas), standardize=False, resampling_target='labels')
    reextracted = masker.fit_transform(str(pet)).ravel()
    np.testing.assert_allclose(reextracted[:90], raw[key], rtol=0, atol=0)
    # Named selection follows the original Hansen study selection where available.
    # Spreng n=3 VAChT is absent in this AAL extraction: use the three available
    # studies explicitly, not an invented fourth input or positional substitution.
    groups = {
        '5-HT1A': [('5HT1a_way_hc36_savli',36)],
        '5-HT1B': [('5HT1b_p943_hc22_savli',22),('5HT1b_p943_hc65_gallezot',65)],
        '5-HT2A': [(key,29)], '5-HT4': [('5HT4_sb20_hc59_beliveau',59)],
        '5-HT6': [('5HT6_gsk_hc30_radhakrishnan',30)], '5-HTT': [('5HTT_dasb_hc100_beliveau',100)],
        'α4β2': [('A4B2_flubatine_hc30_hillmer',30)], 'CB1': [('CB1_omar_hc77_normandin',77)],
        'D1': [('D1_SCH23390_hc13_kaller',13)],
        'D2': [('D2_flb457_hc37_smith',37),('D2_flb457_hc55_sandiego',55)],
        'DAT': [('DAT_fpcit_hc174_dukart_spect',174)],
        'GABA-A': [('GABAa-bz_flumazenil_hc16_norgaard',16)], 'H3': [('H3_cban_hc8_gallezot',8)],
        'M1': [('M1_lsn_hc24_naganawa',24)],
        'mGluR5': [('mGluR5_abp_hc22_rosaneto',22),('mGluR5_abp_hc28_dubois',28),('mGluR5_abp_hc73_smart',73)],
        'MOR': [('MU_carfentanil_hc204_kantonen',204)], 'NET': [('NAT_MRB_hc77_ding',77)],
        'NMDA': [('NMDA_ge179_hc29_galovic',29)],
        'VAChT': [('VAChT_feobv_hc4_tuominen',4),('VAChT_feobv_hc5_bedard_sum',5),('VAChT_feobv_hc18_aghourian_sum',18)]}
    combined = {}
    for label, studies in groups.items():
        x = raw[[s[0] for s in studies]].to_numpy()
        if len(studies)>1:
            x = (x-x.mean(axis=0))/x.std(axis=0)
        combined[label] = np.average(x, axis=1, weights=[s[1] for s in studies])
    combined = pd.DataFrame(combined, index=raw.index)
    maps = pd.DataFrame(minmax(combined), index=raw.index, columns=combined.columns)
    assert maps.shape == (90,19) and np.isfinite(maps).all().all()
    np.testing.assert_allclose(maps.min(),0)
    np.testing.assert_allclose(maps.max(),1)
    raw.to_csv(out/'tracers_aal90_corrected_raw.csv', index_label='region')
    (raw/raw.max()).to_csv(out/'tracers_aal90_corrected_maxscaled.csv', index_label='region')
    combined.to_csv(out/'receptors_aal90_combined_unscaled.csv', index_label='region')
    maps.to_csv(out/'receptors_aal90_19_minmax.csv', index_label='region')
    audit = dict(production_input=str(oldpath), production_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
        raw_indexing='first 90 of 116 nonbackground parcels', historical_bug='entries 1:91 (one-region shift)',
        tracer_checks=checks, groups=groups, normalization='(x-min(x))/(max(x)-min(x)), independently per map',
        five_ht2a_old_new_correlation=float(np.corrcoef(old[key],raw[key])[0,1]),
        verified_reextraction_max_error=float(np.max(abs(reextracted[:90]-raw[key]))),
        remote_hpc_verified=False, production_files_modified=False,
        caveat='VAChT excludes unavailable Spreng n=3; weights are 4,5,18. Negative PET estimates are not literal negative receptor densities. Min-max does not validate PET quantification.')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
    # Preserve alternating L/R AAL order; do not falsely divide hemispheres at 45.
    fig,ax=plt.subplots(figsize=(10,17))
    im=ax.imshow(maps,aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=1)
    ax.set_xticks(range(19),maps.columns,rotation=55,ha='right')
    ax.set_yticks(range(90),maps.index,fontsize=6.5)
    ax.set_title('Receptor architecture across AAL90',loc='left',fontsize=19,pad=20)
    ax.set_xlabel('Receptor / transporter');ax.set_ylabel('AAL region · left/right pairs')
    for edge in np.arange(1.5,90,2):ax.axhline(edge,color='white',lw=.25,alpha=.2)
    fig.colorbar(im,ax=ax,shrink=.35,pad=.03,label='Within-map relative value (min–max)')
    fig.tight_layout()
    for ext in ('png','svg'):fig.savefig(out/f'receptor_heatmap_corrected.{ext}',dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(minmax(old[key]),label='Comparison table',color='#a84c60',lw=1.5)
    ax.plot(maps['5-HT2A'],label='Corrected regional alignment',color='#167b87',lw=1.5)
    ax.set(xlabel='AAL90 region index (0–89)',ylabel='Relative 5-HT2A weight',title='5-HT2A regional alignment audit')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'5ht2a_alignment_audit.png',dpi=220);plt.close(fig)
    # Anatomical slices include subcortical parcels that a cortical surface omits.
    atlas=nib.load(args.atlas); labelsvol=np.asanyarray(atlas.dataobj)
    fig=plt.figure(figsize=(15,16),facecolor='white')
    for j,col in enumerate(maps):
        vol=np.zeros(labelsvol.shape,dtype=np.float32)
        for roi,value in zip(ids[:90],maps[col]):vol[labelsvol==roi]=value
        img=nib.Nifti1Image(vol,atlas.affine)
        ax=fig.add_subplot(7,3,j+1)
        plotting.plot_stat_map(img,display_mode='z',cut_coords=[-12,6,30],axes=ax,figure=fig,
            colorbar=False,cmap='magma',vmin=0,vmax=1,threshold=None,symmetric_cbar=False,
            annotate=False,draw_cross=False,title=col)
    fig.suptitle('AAL90 receptor maps · independently scaled 0–1',fontsize=21,y=.99)
    cax=fig.add_axes([.38,.035,.24,.012])
    scalar=plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap='magma')
    fig.colorbar(scalar,cax=cax,orientation='horizontal',label='Within-map relative value')
    fig.savefig(out/'receptor_brain_slices_corrected.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print(json.dumps({k:v for k,v in audit.items() if k not in ('tracer_checks','groups')},indent=2))


if __name__=='__main__':
    main()
