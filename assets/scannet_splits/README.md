# ScanNet v2 official scene splits

Copied verbatim from https://github.com/ScanNet/ScanNet (master,
`Tasks/Benchmark/scannetv2_{train,val,test}.txt`), fetched 2026-10-14.

| file | scenes | sha256 |
|---|---|---|
| scannetv2_train.txt | 1201 | 96acca299b7855f02824c496b19077904d80996e7ced1bb9f0dac98f7dd4d0c8 |
| scannetv2_val.txt | 312 | d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083 |
| scannetv2_test.txt | 100 | 0214c6a3b1ee516ad653393b0321e7c0394c7662a4b3702eac1ddd7fbc00f7e0 |

Checked against the processed tree on Lustre: `processed_scannet/scans_train`
(1513 scenes) is exactly train + val, and `processed_scannet/scans_test` (100)
is exactly test -- the benchmark's ScanNet scenes. Used by
`ScanNet_Multi(official_split=...)`: train on the 1201, validate / select on
the 312, keep `scans_test` for the benchmark only.
