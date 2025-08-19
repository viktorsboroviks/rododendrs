ref restores to 0
ctx restores to some prev value and the issue propogates from there

kstest - update a
kstest - i: 9
max_pdiff: 0.381
             a_next b_next a_saved b_saved
begin:            0      0       0       0
end:             11     26      11      26
is_end:           0      0       0       0
i:               10     13       8       9
i_vnew:          10     13       8       9
i_max_pdiff:      8      9       5       2
p_step:      0.0909 0.0385  0.0909  0.0385
p:            0.909    0.5   0.727   0.346

correct:

kstest - break
kstest - return
max_pdiff: 0.225
             a_next b_next a_saved b_saved
begin:            0      0       0       0
end:             18     38      18      38
is_end:           1      1       0       0
i:               18     38       5       2
i_vnew:          17     33       5       2
i_max_pdiff:      5      2       4       2
p_step:      0.0556 0.0263  0.0556  0.0263
p:                1      1   0.278  0.0526

wrong:
kstest - break
kstest - return
max_pdiff: 0.208
             a_next b_next a_saved b_saved
begin:            0      0       0       0
end:             18     38      16      32
is_end:           1      1       0       0
i:               18     38      11      31
i_vnew:          17     33      11      31
i_max_pdiff:      8      9       8       9
p_step:      0.0556 0.0263  0.0625  0.0312
p:                1      1   0.688   0.969
