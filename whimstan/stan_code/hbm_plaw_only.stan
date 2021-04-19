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


  real index_mu;
  real<lower=0> index_sigma;
  vector[N_grbs] index_raw;


  real log_K_mu_raw;
  real<lower=0> log_K_sigma;


  vector[N_grbs] log_K_raw; // raw energy flux norm

  //vector[N_grbs] index;

  // vector<lower=0>[N_grbs] nH_mw;

}


transformed parameters{
  vector[N_grbs] index;
  vector[N_grbs] log_K; // log eflux
  vector[N_grbs] K;

  real log_K_mu = log_K_mu_raw - 9;



  // non centered parameterizartion


  index = index_mu + index_raw * index_sigma;

  log_K = log_K_mu + log_K_raw * log_K_sigma;

  K =  pow(10,log_K);


}


model{


  index_raw ~ std_normal();
  log_K_raw ~ std_normal();

  log_K_mu_raw ~ std_normal();
  log_K_sigma ~ std_normal();
  //index ~ normal(-2, 0.5);

  index_mu ~ normal(-2, .1);
  index_sigma ~ std_normal();



  target += reduce_sum(partial_log_like_plaw, all_N, grainsize, N_ene, N_chan, ene_avg, ene_width, mask, n_chans_used, K, index, rsp, exposure, exposure_ratio, counts, bkg );

}
