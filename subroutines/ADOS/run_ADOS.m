addpath("NetworkDOS-master")

generate_ADOS_SBM("SBM_attr", 50, 1)


%generate_ADOS_real("covidflight", 50, 1)

%{
num_trials = 5;
dos_times = zeros(num_trials,1);

for i = 1:num_trials
    dos_times(i) = generate_ADOS_SBM("BA1000", 50, i);
end

mean_dos = mean(dos_times(:));
std_dos = std(dos_times(:));

fprintf("average dos time %f seconds\n", mean_dos);
fprintf("standard deviation of dos time %f seconds\n", std_dos);
%}
