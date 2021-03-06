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
  vector[N_grbs] nH_mw;
  vector[N_ene] precomputed_absorp[N_grbs];
  vector[N_ene] host_precomputed_absorp[N_grbs];
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

  //mw abs is fixed
  vector[N_ene] mw_abs[N_grbs];

  for (n in 1:N_grbs){
    mw_abs[n] = absorption(nH_mw[n], precomputed_absorp[n]);
  }

  // precalculation of energy bounds

  for (n in 1:N_grbs) {

    all_N[n] = n;

  }
}

parameters{

  //vector<upper=0>[N_grbs] index;

  real log_nH_host_mu_raw;

  real<lower=0> log_nH_host_sigma;
  vector[N_grbs] log_nH_host_raw;

  real index_mu;
  real<lower=0> index_sigma;
  vector[N_grbs] index_raw;


  real log_K_mu_raw;
  real<lower=0> log_K_sigma;


  vector[N_grbs] log_K_raw; // raw energy flux norm

  // absori parameter
  real log_n0_whim_raw;
  real log_t_whim_raw;
}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;
  vector[N_grbs] log_nH_host;
  //  vector[N_grbs] nH_host;
  vector[N_grbs] nH_host_norm;
  real log_n0_whim = log_n0_whim_raw -7;
  real log_t_whim = log_t_whim_raw + 7;



  real log_K_mu = log_K_mu_raw - 9;

  // absori
  real n0_whim = pow(10, log_n0_whim);

  // free temp
  matrix[10,26] num;
  real t_whim=pow(10,log_t_whim);

  num = calc_num(spec, t_whim, xi, atomicnumber, sigma, ion);
  for (i in 1:10){
    num[i] = abundance[i]*num[i];
  }

  // non centered parameterizartion

  log_nH_host = log_nH_host_mu_raw + log_nH_host_raw * log_nH_host_sigma;

  index = index_mu + index_raw * index_sigma;

  log_K = log_K_mu + log_K_raw * log_K_sigma;

  K =  pow(10.,log_K);

  nH_host_norm = pow(10.,log_nH_host);



}


model{



  index_raw ~ std_normal();
  log_K_raw ~ std_normal();
  log_nH_host_raw ~ std_normal();

  log_nH_host_mu_raw ~ std_normal();
  log_nH_host_sigma ~ std_normal();

  log_K_mu_raw ~ std_normal();
  log_K_sigma ~ std_normal();


  index_mu ~ normal(-2, .1);
  index_sigma ~ std_normal();



  //absori
  log_n0_whim_raw ~ std_normal();

  log_t_whim_raw ~ std_normal();

  target += reduce_sum(partial_log_like_all, all_N, grainsize, N_ene, N_chan, ene_avg, ene_width, mask, n_chans_used, mw_abs, K, index, n0_whim, num, sum_sigma_interp, nH_host_norm, host_precomputed_absorp, rsp, exposure, exposure_ratio, counts, bkg);

}

generated quantities {

  // vector[N_ene] whim_abs[N_grbs];
  // vector[N_ene] host_abs[N_grbs];
  // vector[N_ene] powerlaw_spectrum[N_grbs];
  // vector[N_chan] predicted_counts[N_grbs];
  // vector[N_ene] source_spectrum[N_grbs];
  // vector[N_ene] integral_flux[N_grbs];

  // for (i in 1:N_grbs){
  //   whim_abs[i] = exp(integrate_absori_precalc(sum_sigma_interp[i], num, N_ene)*n0_whim);
  //   host_abs[i] = absorption(nH_host[i], host_precomputed_absorp[i]);
  //   powerlaw_spectrum[i] = powerlaw_flux(ene_avg[i], K[i], index[i], 0.4, 15);
  //   source_spectrum[i] = powerlaw_spectrum[i] .* whim_abs[i] .* mw_abs[i] .* host_abs[i];

  //   integral_flux[i] = source_spectrum[i] .* ene_width[i];

  //   predicted_counts[i] =  (rsp[i] * integral_flux[i]) * exposure[i];
  // }

}
