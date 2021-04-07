functions {
#include absori.stan
#include tbabs.stan
#include powerlaw.stan
#include cstat.stan
#include partial_log_like.stan
}

data{

  int N_grbs;
  int N_ene;
  int N_chan;
  matrix[N_chan, N_ene] rsp[N_grbs];
  vector[N_grbs] z; //redshift
  vector[N_grbs] exposure_ratio;
  vector[N_ene] ene_avg[N_grbs];
  vector[N_ene] ene_width[N_grbs];
  int n_chans_used[N_grbs];
  vector[N_chan] counts[N_grbs];
  vector[N_chan] bkg[N_grbs];
  int mask[N_grbs,N_chan];
  vector[N_grbs] exposure;

  // absori input
  real xi;
  int atomicnumber[10];
  real ion[10,26,10];
  real sigma[10,26,721];
  vector[721] spec;
  vector[10] abundance;
  matrix[10,26] sum_sigma_interp[N_grbs, N_ene];
}


transformed data{
  int all_N[N_grbs];
  int grainsize = 1;

  // fixed temp at the moment
  matrix[10,26] num;
  real t_whim=pow(10,7);
  vector[N_ene] whim_abs_base[N_grbs];

  num = calc_num(spec, t_whim, xi, atomicnumber, sigma, ion);
  for (i in 1:10){
    num[i] = abundance[i]*num[i];
  }

  for (n in 1:N_grbs){
      whim_abs_base[n] = integrate_absori_precalc(sum_sigma_interp[n], num, N_ene);
  }

  // precalculation of energy bounds

  for (n in 1:N_grbs) {

    all_N[n] = n;

  }
}

parameters{

  //vector<upper=0>[N_grbs] index;
  vector[N_grbs] index;
  vector[N_grbs] log_K_raw; // raw energy flux norm

  // absori parameter
  real log_n0_whim;
  //real log_t_whim;
}


transformed parameters{
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;

  // absori
  real n0_whim = pow(10, log_n0_whim);
  //real t_whim = pow(10, log_t_whim);
  //matrix[10,26] num;


  log_K = log_K_raw -12;
  
  K =  pow(10, log_K);


  // absori - moved to transformed data as temp is fixed at the moment
  //num = calc_num(spec, t_whim, xi, atomicnumber, sigma, ion);
  //for (i in 1:10){
  //  num[i] = abundance[i]*num[i];
  //}
  
}


model{


  index ~ normal(-2, 1);
  log_K_raw ~ normal(0, 3);
  
  //absori
  log_n0_whim ~ normal(-7,1);

  // fix temp for the moment
  // log_t_whim ~ normal(7,1);
 
  target += reduce_sum(partial_log_like_whimonly_t_fixed, all_N, grainsize, N_ene, N_chan, ene_avg, ene_width, mask, n_chans_used, K, index, n0_whim, whim_abs_base, rsp, exposure, exposure_ratio, counts, bkg);

}

generated quantities {

	  vector[N_ene] whim_abs[N_grbs];
	  vector[N_ene] powerlaw_spectrum[N_grbs];
	  vector[N_chan] predicted_counts[N_grbs];
	  vector[N_ene] source_spectrum[N_grbs];
	  vector[N_ene] integral_flux[N_grbs];

	  for (i in 1:N_grbs){
	      whim_abs[i] = exp(whim_abs_base[i]*n0_whim);
	      powerlaw_spectrum[i] = powerlaw_flux(ene_avg[i], K[i], index[i], 0.4, 15);
	      source_spectrum[i] = powerlaw_spectrum[i] .* whim_abs[i];

	      integral_flux[i] = source_spectrum[i] .* ene_width[i];
	      
	      predicted_counts[i] =  (rsp[i] * integral_flux[i]) * exposure[i];
	      }

}

	      