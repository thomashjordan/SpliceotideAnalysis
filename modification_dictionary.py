"""Residue composition table used for every mass calculation in the project.

``aa_comp_custom`` is a copy of pyteomics' standard amino-acid compositions extended with the
custom residue codes that appear in ``modified_sequences`` in the sample sheets (e.g. ``x``
in ``GGCGPDPxVGVILPWA`` for the spliced product). Each code maps to its elemental composition
below. Add new modification codes here.
"""
from pyteomics import mass, auxiliary

# Copy, don't alias: `mass.std_aa_comp` is pyteomics' process-wide standard residue table. Aliasing
# it and adding keys below would silently inject our custom residues (orL, dhA, x, xdh, ...) into
# every other pyteomics user in the process. dict() gives an independent mapping; the Composition
# values are shared by reference, which is fine since they're never mutated.
aa_comp_custom = dict(mass.std_aa_comp)
# we add the custom modifications to the aa_comp dictionary
aa_comp_custom["orL"] = mass.Composition({'C': 5, 'H': 11, 'N': 1, 'O': 1})
aa_comp_custom["dhA"] = mass.Composition({'C': 3, 'H': 3, 'N': 1, 'O': 1})
aa_comp_custom["x"] = mass.Composition({'C': 1, 'O': 1})
aa_comp_custom["xdh"] = mass.Composition({'C': 1, 'O': 1, 'H': -2})
aa_comp_custom["xdhA"] = mass.Composition({'C': 4, 'H': 3, 'N': 1, 'O': 2})
aa_comp_custom["xdhB"] = mass.Composition({'C': 5, 'H': 5, 'N': 1, 'O': 2})
aa_comp_custom["c10ac"] = mass.Composition({'C': 10, 'H': 18, 'O': 1})
aa_comp_custom["c12ac"] = mass.Composition({'C': 12, 'H': 22, 'O': 1})


# we define another dictionary for auxiliary.UnlocalizedModification of different types
'''mod_dict_unlocalized = {
    "acetyl": auxiliary.UnlocalizedModification(name="acetyl", composition=mass.Composition({'C': 2, 'H': 2, 'O': 1}),
                                                classification='modification'),
    "methyl": auxiliary.UnlocalizedModification(name="methyl", composition=mass.Composition({'C': 1, 'H': 2}),
                                               classification='modification'),
    "phospho": auxiliary.UnlocalizedModification(name="phospho", composition=mass.Composition({'H': 1, 'O': 3, 'P': 1}),
                                                classification='modification'),
    "butyryl": auxiliary.UnlocalizedModification(name="butyryl", composition=mass.Composition({'C': 4, 'H': 6, 'O': 1}),
                                                    classification='modification'),
    "hexanoyl": auxiliary.UnlocalizedModification(name="caproyl", composition=mass.Composition({'C': 6, 'H': 10, 'O': 1}),
                                                    classification='modification'),
    "octanoyl": auxiliary.UnlocalizedModification(name="capryryl", composition=mass.Composition({'C': 8, 'H': 14, 'O': 1}),
                                                    classification='modification'),
    "decanoyl": auxiliary.UnlocalizedModification(name="capryl", composition=mass.Composition({'C': 10, 'H': 18, 'O': 1}),
                                                    classification='modification'),
    "lauroyl": auxiliary.UnlocalizedModification(name="lauryl", composition=mass.Composition({'C': 12, 'H': 22, 'O': 1}),
                                                    classification='modification'),
    "myristoyl": auxiliary.UnlocalizedModification(name="myristoyl", composition=mass.Composition({'C': 14, 'H': 26, 'O': 1}),
                                                    classification='modification'),
    "palmitoyl": auxiliary.UnlocalizedModification(name="palmitoyl", composition=mass.Composition({'C': 16, 'H': 30, 'O': 1}),
                                                    classification='modification'),
    "stearoyl": auxiliary.UnlocalizedModification(name="stearoyl", composition=mass.Composition({'C': 18, 'H': 34, 'O': 1}),
                                                    classification='modification'),}
'''
