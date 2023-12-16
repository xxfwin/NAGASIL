for seed in 404 7551 9736 8946 6904
do 

	for G in G_1250_08
	do 
		sbatch sbatch2.sh bprun_gasil.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_ablation/gasil_${G}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_psv.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_ablation/gasildense_${G}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_neg.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_ablation/gasilneg_${G}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_psv_neg.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_ablation/gasildenseneg_${G}_${seed} 
	done

done
