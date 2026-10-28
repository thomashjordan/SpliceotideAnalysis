# rosetta_protocols/

RosettaScripts protocols used to model the designed precursor peptides in the
maturase active site. The scores they produce are the ones correlated against
the measured modification ratios in `plate_analysis_summaries.ipynb` (the
`rosetta_top_47` experiment, via `rosetta_modeling/selected_plp_sequences.xlsx`).

Each protocol is two files: a `.flags` file with the command-line options, and
the `.xml` RosettaScripts protocol that it loads with `-parser:protocol`.

## The two protocols

| Protocol | Refinement after mutation |
|----------|---------------------------|
| `model_peptides` | Full version. A 10-mover block (soft repack → soft min → soft repack → hard min → hard repack → hard min → hard repack → RTmin → RTmin → hard min) run twice via `LoopOver iterations="2"`. |
| `short_model_peptides` | Cheap version, for screening many sequences. One `min` → `prm` (hard repack) → `min`, with repacking restricted to residues within 15 Å of the peptide (`fix_not_neighbor`). Also reports the extra `DesignableResidues` filters used to confirm which positions each task operation actually touched. |

Both share the same setup: `ref2015` for scoring (with
`coordinate_constraint` reweighted to 0.6 in `ref_full`), the peptide library
positions 84A–91A as the design window, coordinate constraints pinning the
ligands (0.01 Å) and the listed active-site side chains (0.5 Å), and the
ligands plus those side chains held fixed during repacking.

Reported per model: `total_score` (constrained and pure), RMSD to the input,
`PackStat`, shape complementarity of the peptide against chains D/X/Y,
per-ligand `EnzScore` for the four ligands (1X–4X), the peptide-only score
(`energy_peptide`), and the peptide SASA.

## Running them

```bash
rosetta_scripts.linuxgccrelease @model_peptides.flags \
    -nstruct <N> \
    -parser:script_vars new_res0=<AA> target0=84A pac0=0 \
                        new_res1=<AA> target1=85A pac1=0 \
                        ... \
                        new_res5=<AA> target5=<pos> pac5=0
```

The six `MutateResidue` movers are parameterised with script variables
(`%%new_res0%%`, `%%target0%%`, `%%pac0%%` … ), so one sequence is modelled per
run and the variables were supplied per job rather than being baked into the
XML.

## Inputs not included here

These are required by the `.flags` files but are not in this repository:

- `plp_ref.pdb.gz` — the input/native complex (maturase + precursor peptide + ligands).
- `LG1.params`, `LG2.params`, `LG3.params`, `LG4.params` — residue-type
  parameters for the four ligands, passed via `-extra_res_fa`.

They will be deposited with the rest of the data; see the main
[README](../README.md#data-availability).

Note also that `-database` and `-parser:protocol` in the `.flags` files are
written for the original machine's layout — point them at your Rosetta
installation and at the `.xml` in this folder.

## Versions

> **TODO:** record the exact Rosetta release/commit used, and the `-nstruct` per
> design, before submission. The protocols themselves only pin the score
> function (`ref2015` / `ref2015_soft`).
