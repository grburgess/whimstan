real pll_no_whim(int [] n_slice,
                 int start,
                 int end,
                 int N_ene,
                 int N_chan,
                 vector[] host_precomputed_absorp,
                 vector[] precomputed_absorp,
                 vector[] ene_avg,
                 vector[] ene_width,
                 int[,] mask,
                 int[] n_chans_used,
                 vector K,
                 vector index,
                 vector nH_host,
                 vector[] mw_abs,
                 //matrix[] rsp ,
		 matrix rmf,
		 vector[] arf,
                 vector exposure,
                 vector exposure_ratio,
                 vector[] counts,
                 vector[] bkg,
                 vector[] log_fact_obs,
                 vector[] log_fact_bkg,
                 vector[] o_plus_b,
                 vector[] alpha_bkg_factor


                 ){

  // host and mw ONLY absorption, fixed MW

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    int n = n_slice[i];



    loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                 bkg[n,mask[n,:n_chans_used[n]]],
                                 ((rmf[n] * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n], 0.4, 15) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n]) .* ene_width[n])[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
                                 exposure_ratio[n],
                                 o_plus_b[n,mask[n,:n_chans_used[n]]],
                                 alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                 log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                 log_fact_bkg[n,mask[n,:n_chans_used[n]]]
                                 );


    // loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
    //                              bkg[n,mask[n,:n_chans_used[n]]],
    //                              ((rsp[n] * ((powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n]) .* ene_width[n])  * exposure[n]))[mask[n,:n_chans_used[n]]],
    //                              exposure_ratio[n],
    //                              o_plus_b[n,mask[n,:n_chans_used[n]]],
    //                              alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_obs[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_bkg[n,mask[n,:n_chans_used[n]]]
    //                              );





  }
  return sum(loglike);

}



real pll_whim(int [] n_slice,
              int start,
              int end,
              int N_ene,
              int N_chan,
              vector[] ene_avg,
              vector[] ene_width,
              int[,] mask,
              int[] n_chans_used,
              vector[] mw_abs,
              vector K,
              vector index,
              real n0,
              matrix num,
              matrix[,] sum_sigma_interp,
              vector nH_host,
              vector[] host_precomputed_absorp,
              matrix[] rsp ,
              vector exposure,
              vector exposure_ratio,
              vector[] counts,
              vector[] bkg,
              vector[] log_fact_obs,
              vector[] log_fact_bkg,
              vector[] o_plus_b,
              vector[] alpha_bkg_factor
              ){

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    // fill the log likelihood array

    int n = n_slice[i];

    loglike[i] =   cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                   bkg[n,mask[n,:n_chans_used[n]]],
                                   (rsp[n] * ((powerlaw_flux(ene_avg[n], K[n], index[n], 0.4, 15) .* exp(integrate_absori_precalc(sum_sigma_interp[n], num, N_ene)*n0) .* mw_abs[n] .* absorption(nH_host[n], host_precomputed_absorp[n])) .* ene_width[n])  * exposure[n])[mask[n,:n_chans_used[n]]],
                                   exposure_ratio[n],
                                   o_plus_b[n,mask[n,:n_chans_used[n]]],
                                   alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                   log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                   log_fact_bkg[n,mask[n,:n_chans_used[n]]]
                                   );

  }

  return sum(loglike);

}
