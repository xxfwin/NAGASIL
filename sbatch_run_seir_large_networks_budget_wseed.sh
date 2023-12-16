for seed in 0 7551 9736 
do 

	for G in G_2500_08
	do 
		sbatch sbatch1.sh bprun_random.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/random_${G}_${seed} 
		sbatch sbatch1.sh bprun_maxinf.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/maxinf_${G}_${seed} 
		sbatch sbatch1.sh bprun_maxdef.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/maxdef_${G}_${seed} 
		sbatch sbatch2.sh bprun_gasil.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/gasil_${G}_${seed} 
		sbatch sbatch3.sh bprun_sidp.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/sidp_${G}_${seed} 
		sbatch sbatch3.sh bprun_dqnfsp.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/dqnfsp_${G}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_psv_neg.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_bignet/gasildenseneg_${G}_${seed} 
	done

done
