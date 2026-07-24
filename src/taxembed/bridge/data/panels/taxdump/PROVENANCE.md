# Taxdump Provenance

## Source

NCBI new_taxdump (names, nodes, merged, delnodes).

Download URL: https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz

## Copy details

- Copied: 2026-06-17
- Copied from: an ephemeral local download of new_taxdump.tar.gz
- Original file mtime: 2026-06-09

## SHA256 hashes

```
names.dmp:     8cbc866448d395328eca3f5863a9cdad63042ded1ab3d203d52d40dca09f533c
nodes.dmp:     98067418cf9653bac0961da13c905147ca1fcc30cb4fd94e6cf6031a385b2a77
merged.dmp:    429ccbfde3dcf202dac3431babf0720485868d57eccc45687a049bef6b1ab17e
delnodes.dmp:  3752d1f899081f309cfb24fba8d8266de91fe0d9d44e16edf69e4aae342cd8a3
```

## Notes

The .dmp files are excluded by the `.gitignore` in this directory (`*.dmp`) and are NOT committed
to the repository. They are regenerable at any time by re-downloading from the NCBI URL above.

`config.TAXDUMP_DIR` in the `taxembed.bridge` package points to this directory. The hand-rolled
.dmp resolver and the species-resolution freeze read from this path.
