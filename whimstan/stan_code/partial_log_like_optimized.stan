real pll_no_whim(array[] int  n_slice,
                 int start,
                 int end,
                 int N_ene,
                 int N_chan,
                 array[] vector host_precomputed_absorp,
                 array[] vector precomputed_absorp,
                 array[] vector ene_avg,
                 array[] vector ene_width,
                 array[,] int mask,
                 array[] int n_chans_used,
                 vector K,
                 vector index,
                 vector nH_host,
                 array[] vector mw_abs,
                 matrix rmf,
                 array[] vector arf,
                 vector exposure,
                 vector exposure_ratio,
                 array[] vector counts,
                 array[] vector bkg,
                 array[] vector log_fact_obs,
                 array[] vector log_fact_bkg,
                 array[] vector o_plus_b,
                 array[] vector alpha_bkg_factor,
                 array[] vector zero_mask
                 ){

  // host and mw ONLY absorption, fixed MW

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    int n = n_slice[i];

    vector[N_chan] input =  (rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]));

    // loglike[i] = cstat_optimized_vec(counts[n,mask[n,:n_chans_used[n]]],
    //                                  bkg[n,mask[n,:n_chans_used[n]]],
    //                                  input[mask[n,:n_chans_used[n]]] * exposure[n] * K[n],
    //                                  exposure_ratio[n],
    //                                  o_plus_b[n,mask[n,:n_chans_used[n]]],
    //                                  alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
    //                                  log_fact_obs[n,mask[n,:n_chans_used[n]]],
    //                                  log_fact_bkg[n,mask[n,:n_chans_used[n]]],
    //                               zero_mask[n,mask[n,:n_chans_used[n]]]
    //                                  );



    loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                 bkg[n,mask[n,:n_chans_used[n]]],
                                 input[mask[n,:n_chans_used[n]]] * exposure[n] * K[n],
                                 exposure_ratio[n],
                                 o_plus_b[n,mask[n,:n_chans_used[n]]],
                                 alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                 log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                 log_fact_bkg[n,mask[n,:n_chans_used[n]]]
                                 );




  }
  return sum(loglike);

}



real pll_whim(array[] int  n_slice,
              int start,
              int end,
              int N_ene,
              int N_chan,
              array[] vector ene_avg,
              array[] vector ene_width,
              array[,] int mask,
              array[] int n_chans_used,
              array[] vector mw_abs,
              vector K,
              vector index,
              real n0,
              vector num,
              array[] matrix sum_sigma_interp,
              vector nH_host,
              array[] vector host_precomputed_absorp,
              //array[] matrix rsp ,
              matrix rmf,
              array[] vector arf,
              vector exposure,
              vector exposure_ratio,
              array[] vector counts,
              array[] vector bkg,
              array[] vector log_fact_obs,
              array[] vector log_fact_bkg,
              array[] vector o_plus_b,
              array[] vector alpha_bkg_factor,
              array[] vector zero_mask
              ){

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;



  for (i in 1:slice_length){

    // fill the log likelihood array

    int n = n_slice[i];

    // computing the WHIM absorption
    vector[N_ene] whim_abs = exp(-n0 *(sum_sigma_interp[n] * num) );

    vector[N_chan] input = (rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* whim_abs .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))


      loglike[i] = cstat_optimized_vec(counts[n,mask[n,:n_chans_used[n]]],
                                       bkg[n,mask[n,:n_chans_used[n]]],
                                       input[mask[n,:n_chans_used[n]]] * exposure[n] * K[n],
                                       exposure_ratio[n],
                                       o_plus_b[n,mask[n,:n_chans_used[n]]],
                                       alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                       log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                       log_fact_bkg[n,mask[n,:n_chans_used[n]]],
                                       zero_mask[n,mask[n,:n_chans_used[n]]]
                                       );





  }

  return sum(loglike);

}
