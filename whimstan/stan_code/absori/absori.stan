matrix calc_num(vector spec, real temp, real xi, int[] atomicnumber, real[,,] sigma, real[,,] ion){

    real mult;
    real ratsum;
    int Ne;
    real intgral;
    real e1;
    real e2;
    real arec;
    real z2;
    real y;
    real s;
    
    real t4=0.0001*temp;
    real tfact=1.033e-3/sqrt(t4);
    real xil=log(xi);

    int num_energy_base=size(sigma[1,1]);
    int num_atomicnumber=size(atomicnumber);
    int max_atomicnumber=max(atomicnumber);
    vector[max_atomicnumber] mul;
    vector[max_atomicnumber] ratio;
    matrix[num_atomicnumber, max_atomicnumber] num;
    for (i in 1:10){
        for (j in 1:26){
            num[i,j]=0.0;
        }
    }

    for (i in 1:num_atomicnumber){
        mult = 0.0;
        ratsum = 0.0;

        Ne = atomicnumber[i];
        mul = rep_vector(0.0, Ne);
        ratio = rep_vector(0.0, Ne);

        for (j in 1:Ne){
            intgral = 0.0;
            for (k in 1:num_energy_base){
                intgral += sigma[i,j,k]*spec[k];
                }
            if (j<Ne){
                e1 = exp(-ion[i,j,5]/t4);
                e2 = exp(-ion[i,j,7]/t4);
                arec = (ion[i,j,2]*pow(t4, -ion[i,j,3])+
                        ion[i,j,4]*pow(t4, -1.5)*
                        e1*(1.0+ion[i,j,6]*e2));
                }
            else {
                z2 = pow(Ne,2.0);
                y = 15.8*z2/t4;
                arec = tfact*z2*(1.735+log(y)+1.0/(6.*y));
                }
            ratio[j] = log(3.2749e-6*intgral/arec);
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


// precalc sigma interpolation for all z we need => 0.02 z steps up to z=max(z) of all GRBs

vector integrate_absori(real z, real n0, matrix num, vector abundance, matrix[,] sigma_interp, int num_e_edges){

       int num_ein = num_e_edges;
       vector[num_ein] taus=rep_vector(0.0, num_ein);
       vector[num_ein] xsec;

       //int num_atomicnumber=size(atomicnumber);
       //int max_atomicnumber=max(atomicnumber);
       //matrix[num_atomicnumber, max_atomicnumber] num = calc_num(spec, temp, xi, atomicnumber, sigma, ion);

       int nz;
       real zsam;
       real z1;
       real zf;

       real omegam=0.3;
       real omegal=0.7;
       real h0=70;
       real c=2.99792458e5;
       real cmpermpc=3.08568e24;

       // how many 0.02 z shells do we need?
       nz=0;
       while (nz*0.02<z){
             nz+=1;
       }

       for (i in 1:nz){
           // "slab approximation" in this z "shell"
           // thickness of the "shell" (only different in the last shell)
           if (i==nz){
              zsam = i*0.02-z;
              }
           else {
              zsam = 0.02;
           }
           z1 = (i*0.02-0.01)+1.0;
           zf = (pow(z1,2)/sqrt(omegam*pow(z1,3)+omegal));
           zf *= zsam*c*n0*cmpermpc/h0;

           for (j in 1:num_ein){
               //print(num.*sigma_interp[i,j]);
               //print(sum(num.*sigma_interp[i,j]));
               xsec[j] = sum(num.*sigma_interp[i,j])*6.6e-5*1e-22;
           }
           // if this is the last shell we have to check
           // "how much" of this shell we really need
           taus += xsec*zf;

           }

       return exp(-taus);
}
