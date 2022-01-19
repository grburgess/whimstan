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
                 array[] vector alpha_bkg_factor
                 ){

  // host and mw ONLY absorption, fixed MW

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    int n = n_slice[i];


    loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                 bkg[n,mask[n,:n_chans_used[n]]],
                                 ((rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
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
	      // array[] vector whim_abs,
	      //array[] vector whim_abs,
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
              array[] vector alpha_bkg_factor
              ){

  int slice_length = num_elements(n_slice);

  vector[slice_length] loglike;

  for (i in 1:slice_length){

    // fill the log likelihood array

    int n = n_slice[i];

    // vector[N_ene] source_spectrum = powerlaw_flux(ene_avg[n], index[n]) .* whim_abs[n] .* mw_abs[n] .* absorption(nH_host[n], host_precomputed_absorp[n]);


    // vector[N_chan] predicted_counts =  (rmf * ( arf[n]  .* source_spectrum .* ene_width[n]));




    // loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
    //                              bkg[n,mask[n,:n_chans_used[n]]],
    //                              (predicted_counts[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
    //                              exposure_ratio[n],
    //                              o_plus_b[n,mask[n,:n_chans_used[n]]],
    //                              alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_obs[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_bkg[n,mask[n,:n_chans_used[n]]]
    //                              );




    // loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
    //                              bkg[n,mask[n,:n_chans_used[n]]],
    //                              ((rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* whim_abs[(n-1) * N_ene +1 : n*N_ene ] .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
    //                              exposure_ratio[n],
    //                              o_plus_b[n,mask[n,:n_chans_used[n]]],
    //                              alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_obs[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_bkg[n,mask[n,:n_chans_used[n]]]
    //                              );

    // loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
    //                              bkg[n,mask[n,:n_chans_used[n]]],
    //                              ((rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* whim_abs[n] .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
    //                              exposure_ratio[n],
    //                              o_plus_b[n,mask[n,:n_chans_used[n]]],
    //                              alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_obs[n,mask[n,:n_chans_used[n]]],
    //                              log_fact_bkg[n,mask[n,:n_chans_used[n]]]
    //                              );
    loglike[i] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                 bkg[n,mask[n,:n_chans_used[n]]],
                                 ((rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* exp(- integrate_absori_vec4(num, sum_sigma_interp[n]* n0 )) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
                                 exposure_ratio[n],
                                 o_plus_b[n,mask[n,:n_chans_used[n]]],
                                 alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                 log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                 log_fact_bkg[n,mask[n,:n_chans_used[n]]]
                                 );





  }

  return sum(loglike);

}


real pll_whim_test(array[] matrix sum_sigma_interp,
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
		   //              array[] matrix sum_sigma_interp,
	      // array[] vector whim_abs,
	      //array[] vector whim_abs,
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
              array[] vector alpha_bkg_factor
              ){

  int slice_length = end - start;

  vector[slice_length] loglike;

  int local_itr = 0; // keep track of the slice size

  for (n in start:end){

    // fill the log likelihood array


    loglike[local_itr] = cstat_optimized(counts[n,mask[n,:n_chans_used[n]]],
                                 bkg[n,mask[n,:n_chans_used[n]]],
                                 ((rmf * ( arf[n] .*  powerlaw_flux(ene_avg[n], index[n]) .* exp(- integrate_absori_vec4(num, sum_sigma_interp[local_itr]* n0 )) .* absorption(nH_host[n], host_precomputed_absorp[n]) .* mw_abs[n] .* ene_width[n]))[mask[n,:n_chans_used[n]]]) * exposure[n] * K[n],
                                 exposure_ratio[n],
                                 o_plus_b[n,mask[n,:n_chans_used[n]]],
                                 alpha_bkg_factor[n,mask[n,:n_chans_used[n]]],
                                 log_fact_obs[n,mask[n,:n_chans_used[n]]],
                                 log_fact_bkg[n,mask[n,:n_chans_used[n]]]
                                 );





  }

  return sum(loglike);

}
