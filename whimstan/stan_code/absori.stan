matrix calc_num(//vector spec,
                real temp,
                real xi,
                array[] int atomicnumber,
                //array[,,] real sigma,
                array[,,] real ion,
                matrix zero_matrix,
                vector zero_vector,
                array[] vector intgral,
                int num_energy_base,
                int num_atomicnumber,
                int max_atomicnumber

                ){

  real mult;
  real ratsum;
  int Ne;
  //  real intgral;
  real e1;
  real e2;
  real arec;
  real z2;
  real y;
  real s;

  real t4=0.0001*temp;
  real tfact=1.033e-3/sqrt(t4);
  real xil=log(xi);

  vector[max_atomicnumber] mul;
  vector[max_atomicnumber] ratio;
  matrix[num_atomicnumber, max_atomicnumber] num = zero_matrix;


  for (i in 1:num_atomicnumber){
    mult = 0.0;
    ratsum = 0.0;

    Ne = atomicnumber[i];
    mul = zero_vector;
    ratio = zero_vector;

    for (j in 1:Ne){

      if (j<Ne){
        e1 = exp(-ion[i,j,5]/t4);
        e2 = exp(-ion[i,j,7]/t4);
        arec = (ion[i,j,2]*pow(t4, -ion[i,j,3])+
                ion[i,j,4]*pow(t4, -1.5)*
                e1*(1.0+ion[i,j,6]*e2));
      }
      else {
        z2 =square(Ne);
        y = 15.8*z2/t4;
        arec = tfact*z2*(1.735+log(y)+1.0/(6.*y));
      }
      ratio[j] = log(3.2749e-6*intgral[i][j]/arec);
      ratsum += ratio[j];
      mul[j] = ratsum + j*xil;
      if (mul[j]>mult){
        mult = mul[j];
      }
    }
    s = 0.0;
    for (j in 1:Ne){
      mul[j] -= mult;
      s += exp(mul[j]);
    }

    s += exp(-mult);

    num[i,1] = -mult-log(s);
    for (j in 2:Ne){
      num[i,j] = num[i,j-1]+ratio[j-1]+xil;
    }
    for (j in 1:Ne){
      num[i,j] = exp(num[i,j]);
    }
  }

  return num;

}


vector calc_num_vec(//vector spec,
                    real temp,
                    real xi,
                    array[] int atomicnumber,
                    //array[,,] real sigma,
                    array[,,] real ion,
		    vector zero_matrix,
                    vector zero_vector,
                    array[] vector intgral,
                    int num_energy_base,
                    int num_atomicnumber,
                    int max_atomicnumber,
                    int num_size

                    ){

  real mult;
  real ratsum;
  int Ne;
  //  real intgral;
  real e1;
  real e2;
  real arec;
  real z2;
  real y;
  real s;

  real t4=0.0001*temp;
  real tfact=1.033e-3/sqrt(t4);
  real xil=log(xi);

  vector[max_atomicnumber] mul;
  vector[max_atomicnumber] ratio;

  vector[num_size] num = zero_matrix;

  for (i in 1:num_atomicnumber){
    mult = 0.0;
    ratsum = 0.0;

    Ne = atomicnumber[i];
    mul = zero_vector;
    ratio = zero_vector;

    for (j in 1:Ne){

      if (j<Ne){
        e1 = exp(-ion[i,j,5]/t4);
        e2 = exp(-ion[i,j,7]/t4);
        arec = (ion[i,j,2]*pow(t4, -ion[i,j,3])+
                ion[i,j,4]*pow(t4, -1.5)*
                e1*(1.0+ion[i,j,6]*e2));
      }
      else {
        z2 =square(Ne);
        y = 15.8*z2/t4;
        arec = tfact*z2*(1.735+log(y)+1.0/(6.*y));
      }
      ratio[j] = log(3.2749e-6*intgral[i][j]/arec);
      ratsum += ratio[j];
      mul[j] = ratsum + j*xil;
      if (mul[j]>mult){
        mult = mul[j];
      }
    }
    s = 0.0;
    for (j in 1:Ne){
      mul[j] -= mult;
      s += exp(mul[j]);
    }

    s += exp(-mult);

    //num[i,1] = -mult-log(s);

    num[(i-1)*max_atomicnumber +1] = -mult-log(s);



    for (j in 2:Ne){
      num[(i-1)*max_atomicnumber +j] = num[(i-1)*max_atomicnumber +j-1]+ratio[j-1]+xil;
    }

    for (j in 1:Ne){
      num[(i-1)*max_atomicnumber +j] = exp(num[(i-1)*max_atomicnumber +j]);
    }
  }

  return num;

}




// precalc sigma interpolation for all z we need => 0.02 z steps up to z=max(z) of all GRBs

matrix log_absori_shells(int nz_shells,
                         real zshell_thickness,
                         real n0,
                         matrix num,
                         array[,] matrix sigma_interp,
                         int num_e_edges,
                         array[] int atomicnumber){

  matrix[nz_shells,num_e_edges] taus=rep_matrix(0.0, nz_shells, num_e_edges);

  real zsam;
  real z1;
  real zf;

  for (i in 1:nz_shells){
    // "slab approximation" in this z "shell"
    z1 = ((i-0.5)*zshell_thickness)+1.0;
    // zf from z integral (see eq. 1 in arxiv 2102.02530)
    zf = ((z1*z1)/sqrt(omegam()*(z1*z1*z1)+omegal() ));

    // for every energy
    for (j in 1:num_e_edges){
      taus[i,j] = sum(num.*sigma_interp[i,j])*zf;
    }
  }
  taus = zshell_thickness*c()*n0*cmpermpc()/h0()*6.6e-5*1e-22*taus;

  return taus;
}

//if t is fixed we can use this to precalc most of it

vector integrate_absori_precalc(array[] matrix sum_sigma_interp,
                                matrix num,
                                int num_e_edges){



  vector[num_e_edges] taus;



  for (j in 1:num_e_edges){

    profile("inside"){
      taus[j] = -sum(sum_sigma_interp[j] .* num);
    }
  }



  return taus;
}





vector integrate_absori_vec(vector num,
                            vector theta, // not used
                            data array[] real x_r,
                            data array[] int x_i // not used
                            ){

  int N_e_edges = 2400;

  vector[N_e_edges] taus;

  for (n in 1:N_e_edges){

    profile("inside_vec"){

      for (i in 1:10) {


        for (j in 1:26){

	  //print(n,i,j);
	  //print( (n-1)*10*26 + (i-1)*26 + j);
          taus[n] += -(x_r[(n-1)*10*26 + (i-1)*26 + j] * num[(i-1)*26 + j]);
	  //taus[n] = 0;
        }

      }
    }
  }



  return taus;
}













vector integrate_absori(array[] matrix sum_sigma_interp,
                        matrix num,
                        real n0,
                        int num_e_edges){
  vector[num_e_edges] taus;
  for (j in 1:num_e_edges){
    taus[j] = -n0*sum(sum_sigma_interp[j].*num);
  }
  return exp(taus);
}


vector integrate_absori2(real z,
                         matrix logabso_shells,
                         real zshell_thickness,
                         int num_e_edges,
                         int n_spectra){

  vector[num_e_edges] taus;
  int nz;
  real frac;

  nz=0;
  while ((nz+1)*zshell_thickness<z){
    nz+=1;
  }
  frac = ((nz+1)*zshell_thickness-z)/z;

  for (j in 1:num_e_edges){
    taus[j] = -(sum(logabso_shells[:nz,j])+frac*logabso_shells[nz+1,j]);
  }
  return exp(taus);
}
