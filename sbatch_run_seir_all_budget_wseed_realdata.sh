mkdir -p result/epidemicdecayplb20_fixedI_realdata/
for G in facebook_564_r2.G facebook_453_r2.G 
do 
	for seed in 404 7551 9736 8946 6904
	do 
		for duration in 0.5 0.75 1.0 1.25 1.5
		do
			sbatch sbatch1.sh bprun_random.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/random_${G}_d_${duration}_${seed} 
			sbatch sbatch1.sh bprun_maxinf.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/maxinf_${G}_d_${duration}_${seed} 
			sbatch sbatch1.sh bprun_maxdef.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/maxdef_${G}_d_${duration}_${seed} 
			sbatch sbatch2.sh bprun_gasil.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/gasil_${G}_d_${duration}_${seed} 
			sbatch sbatch2.sh bprun_sidp.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/sidp_${G}_d_${duration}_${seed} 
			sbatch sbatch2.sh bprun_dqnfsp.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/dqnfsp_${G}_d_${duration}_${seed} 
			sbatch sbatch2.sh bprun_gasil_dense_psv_neg.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep $duration --stagenumber 20  --endtime $(echo "$duration*20 + 10" | bc) --iteration 1000  --budget 20  --resultdir result/epidemicdecayplb20_fixedI_realdata/gasildenseneg_${G}_d_${duration}_${seed} 
		done

		for budget in 10 15 25 30
		do 
			sbatch sbatch1.sh bprun_random.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/random_${G}_budget_${budget}_${seed} 
			sbatch sbatch1.sh bprun_maxinf.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/maxinf_${G}_budget_${budget}_${seed}
			sbatch sbatch1.sh bprun_maxdef.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/maxdef_${G}_budget_${budget}_${seed}
			sbatch sbatch2.sh bprun_gasil.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/gasil_${G}_budget_${budget}_${seed}
			sbatch sbatch2.sh bprun_sidp.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1  --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/sidp_${G}_budget_${budget}_${seed}
			sbatch sbatch2.sh bprun_dqnfsp.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/dqnfsp_${G}_budget_${budget}_${seed}
			sbatch sbatch2.sh bprun_gasil_dense_psv_neg.py --graphpath env/${G}  --seed $seed  --starttime 5  --timestep 1 --stagenumber 20  --endtime 30 --iteration 1000  --budget $budget  --resultdir result/epidemicdecayplb20_fixedI_realdata/gasildenseneg_${G}_budget_${budget}_${seed}
		done
	done
done
