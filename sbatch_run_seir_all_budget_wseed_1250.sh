for seed in 404 7551 9736 8946 6904
do 
	for duration in 0.5 0.75 1.25 1.5
	do
		sbatch sbatch1.sh bprun_random.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/random_d_${duration}_${seed} 
		sbatch sbatch1.sh bprun_maxinf.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/maxinf_d_${duration}_${seed} 
		sbatch sbatch1.sh bprun_maxdef.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/maxdef_d_${duration}_${seed} 
		sbatch sbatch2.sh bprun_gasil.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/gasil_d_${duration}_${seed} 
		sbatch sbatch3.sh bprun_sidp.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/sidp_d_${duration}_${seed} 
		sbatch sbatch3.sh bprun_dqnfsp.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/dqnfsp_d_${duration}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_psv_neg.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/gasildenseneg_d_${duration}_${seed} 
	done

	for G in  G_250_08 G_500_08 G_750_08 G_1000_08 G_1250_08 G_1250_07 G_1250_075 G_1250_085 G_1250_09
	do 
		sbatch sbatch1.sh bprun_random.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/random_${G}_${seed} 
		sbatch sbatch1.sh bprun_maxinf.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/maxinf_${G}_${seed} 
		sbatch sbatch1.sh bprun_maxdef.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/maxdef_${G}_${seed} 
		sbatch sbatch2.sh bprun_gasil.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/gasil_${G}_${seed} 
		sbatch sbatch3.sh bprun_sidp.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/sidp_${G}_${seed} 
		sbatch sbatch3.sh bprun_dqnfsp.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/dqnfsp_${G}_${seed} 
		sbatch sbatch3.sh bprun_gasil_dense_psv_neg.py --graphpath env/b$G  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_1250/gasildenseneg_${G}_${seed} 
	done

	for budget in 10 15 25 30
	do 
		sbatch sbatch1.sh bprun_random.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/random_budget_${budget}_${seed} 
		sbatch sbatch1.sh bprun_maxinf.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/maxinf_budget_${budget}_${seed}
		sbatch sbatch1.sh bprun_maxdef.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/maxdef_budget_${budget}_${seed}
		sbatch sbatch2.sh bprun_gasil.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/gasil_budget_${budget}_${seed}
		sbatch sbatch3.sh bprun_sidp.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1  --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/sidp_budget_${budget}_${seed}
		sbatch sbatch3.sh bprun_dqnfsp.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/dqnfsp_budget_${budget}_${seed}
		sbatch sbatch3.sh bprun_gasil_dense_psv_neg.py --graphpath env/bG_1250_08  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_1250/gasildenseneg_budget_${budget}_${seed}
	done


done
