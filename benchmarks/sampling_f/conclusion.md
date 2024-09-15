According to performed benchmark, more effective sampling can be selected:

if sampling_part <= 0.6 -> use sample_in
else                    -> use sample_out

where:

sampling_part = sample_n/population_n
sample_in - sampling by adding samples to set
sample_out - sampling by removing samples from set
