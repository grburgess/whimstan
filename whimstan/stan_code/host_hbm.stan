functions {
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
  vector[N_grbs] exposure_ratio;
  vector[N_ene] precomputed_absorp[N_grbs];
  vector[N_ene] host_precomputed_absorp[N_grbs];
  vector[N_ene] ene_avg[N_grbs];
  vector[N_ene] ene_width[N_grbs];
  int n_chans_used[N_grbs];
  vector[N_chan] counts[N_grbs];
  vector[N_chan] bkg[N_grbs];
  int mask[N_grbs,N_chan];
  vector[N_grbs] exposure;

}


transformed data{
  int all_N[N_grbs];
  int grainsize = 1;


  // precalculation of energy bounds

  for (n in 1:N_grbs) {

    all_N[n] = n;

  }
}

parameters{

  vector<upper=0>[N_grbs] index;
  vector[N_grbs] log_K;

  real log_nH_host_mu;
  real<lower=0> log_nH_host_sigma;

  
  vector[N_grbs] log_nH_host_raw;
  
  // vector<lower=0>[N_grbs] nH_mw;

}


transformed parameters{
  vector<lower=0>[N_grbs] K;
  vector[N_grbs] log_nH_host;
  vector<lower=0>[N_grbs] nH_host;

  log_nH_host = log_nH_host_mu + log_nH_host_raw * log_nH_host_sigma;

  
  K =  exp(log_K);
  nH_host = exp(log_nH_host);

}


model{


  index ~ normal(-2,1);
  log_K ~ normal( log(1e-4), 1);
  log_nH_host_raw ~ std_normal();

  log_nH_host_mu = normal(1, 1);
  log_nH_host_sigma = normal(0, 1);
  

  target += reduce_sum(partial_log_like, all_N, grainsize, N_ene, host_precomputed_absorp, precomputed_absorp, ene_avg, ene_width, mask, n_chans_used, K, index, nH_host, nH_mw, rsp, exposure, exposure_ratio, counts, bkg );

}
