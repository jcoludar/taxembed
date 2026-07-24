# SP-Metazoa stage-1 panel build log

- UniProt release: **2026_02**  ·  query: `reviewed:true AND taxonomy_id:33208`
- PCA_DIM (reps): 64  ·  EC level: 3

## Per-filter counts (§4.1–4.6)
- parsed (reviewed Metazoa): 110691
- single-Pfam-family: 63122
- coverage filter: 63122  **(NOT ENFORCED — `xref_pfam` carries no domain coords; reported no-op, spec §4.3/B4)**
- taxid-resolved into embedding: 62982  (dropped 140)
- present in per-protein.h5: 62981  (missing 1)
- **join-coverage = 1.0000**  (gate §6.5: ≥ 0.98 → PASS)

## Resolution-drop order bias (§4.4)
| order | dropped | kept |
|---|---:|---:|
| Primates | 4 | 15310 |
| Rodentia | 3 | 14986 |
| Artiodactyla | 0 | 5586 |
| Diptera | 0 | 4262 |
| Anura | 2 | 3771 |
| Rhabditida | 0 | 2952 |
| Cypriniformes | 3 | 2227 |
| Squamata | 1 | 2034 |
| Galliformes | 0 | 1392 |
| Carnivora | 0 | 1027 |
| Araneae | 25 | 898 |
| Neogastropoda | 0 | 807 |
| Scorpiones | 0 | 780 |
| Lagomorpha | 0 | 617 |
| Lepidoptera | 0 | 476 |
| Actiniaria | 9 | 338 |
| Hymenoptera | 5 | 320 |
| Salmoniformes | 1 | 321 |
| Perissodactyla | 2 | 287 |
| Blattodea | 39 | 226 |
| Decapoda | 1 | 182 |
| Tetraodontiformes | 0 | 159 |
| Perciformes | 0 | 151 |
| Chiroptera | 0 | 145 |
| Camarodonta | 0 | 124 |
| …(+207 more orders) | | |

## Crossing gate-scan (§4.7 / §6.1)
- Pfam: 170 crossed families (of 4447)  | floor M=50, raw≥4, eff≥3
- Pfam nesting: 1−H(tax|func)=0.232  1−H(func|tax)=0.137  (both < 0.3? True)
- Pfam within-family clade-signal MI range (§4.7a, reported): 0.000–1.871 (median 0.403)
- **Pfam acceptance (§6.1): ACCEPTED** (enough_families=True [≥15], nesting_ok=True)

- EC: 49 crossed EC3 families (of 154)
- EC nesting: 1−H(tax|func)=0.164  1−H(func|tax)=0.121
- **EC acceptance: ACCEPTED**
- **NMI(Pfam, EC3) on 6447 shared proteins = 0.875**  (independent iff < 0.7 → REDUNDANT/NA)

## Notes
- Thresholds are PRE-COMMITTED (§6.1, anti-p-hacking): a failing panel is a real negative, not a retune trigger.
- Exclusions (if any) are mechanical-category only (§4.8/B2); the headline must AGREE between the post-exclusion and zero-exclusion (`_full`) twin in E2.
