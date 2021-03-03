vector absori_opacity_free_T_n0(vector energybounds, real t4, real n0, real z,
       vector abund, vector ionizing_spec, real xil, vector atomicnumber,
       vector energy_base, real sigma[,,], real ion[,,]){

       real tfact = 1.033E-3/sqrt(t4);
       real mult;
       real ratsum;
       real e1;
       real e2;
       real arec;
       real z2;
       real y;
       real ratsum;
       real s;
       int Ne;
       real xp;
       real re;
       real rp;
       real rp1;
       real rp2;
       int num_energy_base=size(energy_base);
       real intgral;

       int num_atomicnumber = size(atomicnumber);
       int max_atomicnumber = max(atomicnumber);
       vector[max_atomicnumber] mul;
       vector[max_atomicnumber] ratio;
       real EnergyMax = energy_base[-1];
       real EnergyMin = energy_base[0];


       real num_ein = size(energybounds)-1;
       real factor = z/(2.0*511.0);

       vector[num_ein] energy = (energybounds[2:]+energybounds[1:num_ein]) * factor;

       vector[num_ein] opactiy=rep_vector(0.0, num_ein);

       for (i in 1:num_atomicnumber){
            mult = 0.0;
            ratsum = 0.0;
            # number of ionization states
            Ne = atomicnumber[i];
            mul = rep_vector(0.0, max_atomicnumber);
            ratio = rep_vector(0.0, max_atomicnumber);

            # for every ionization state
            for (j in 1:Ne){
                intgral = 0.0;
                for (k in 1:num_energy_base){
                    intgral += self._sigma[i,j,k]*ionizing_spec[k];
                    }
                if (j<Ne-1){
                    e1 = exp(-ion[i,j,4]/t4);
                    e2 = exp(-ion[i,j,6]/t4);
                    arec = (ion[i,j,1]*pow(t4, -ion[i,j,2])+
                            ion[i,j,3]*pow(t4, -1.5)*
                            e1*(1.0+ion[i,j,5]*e2));
                    }
                else {
                    z2 = atomicnumber[i]**2;
                    y = 15.8*z2/t4;
                    arec = tfact*z2*(1.735+log(y)+1/(6.*y));
                    }
                ratio[j] = log(3.2749e-6*intgral/arec);
                ratsum += ratio[j];

                mul[j] = ratsum + (j+1)*xil;
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

            num[i,0] = -mult-log(s);
            for (j in 1:Ne){
                num[i,j] = num[i,j-1]+ratio[j-1]+xil;
                }
            for (j in 1:Ne){
                num[i,j] = exp(num[i,j]);
                }
        }

        for (ie in 1:num_ein){
            xp = energy[ie]*5.11e5;
            if (xp >= EnergyMax){
                re = pow((xp/EnergyMax),-3.0);
                for (i in 1:num_atomicnumber){
                    rp = abund[i]*re;
                    for (j in 1:atomicnumber[i]){
                        opacity[ie] += num[i,j]*sigma[i,j,720]*rp;
                        }
                    }
                }
            else if (xp < EnergyMin){
                opacity[ie] = 1.0e20;
                }
            else{
                # Pick the closest energy bin
                while (energy_base[k]<xp){
                    k+=1;
                    }
                re = (xp-energy_base[k-1])/(energy_base[k]-energy_base[k-1]);
                for (i in 1:num_atomicnumber){
                    rp1 = abund[i] * re;
                    rp2 = abund[i] * (1.0 - re);
                    for (j in 1:atomicnumber[i]){
                         opacity[ie] += num[i,j]*(rp1*sigma[i,j,k] +
                                                        rp2*sigma[i,j,k-1]);
                        }
                    }
                }
            }

        opacity *= 6.6e-5;
        return opacity;
}

vector integrate_absori_opacity_free_T_n0(vector energy, real t4, real n0, real z,
       vector abund, vector spec, real xil, vector atomicnumber,
       vector energy_base, real sigma[,,], real ion[,,]){

       real num_ein = size(energybounds)-1;
       vector[num_ein] opacity_int = rep_vector(0.0, num_ein);
       vector[num_ein] opacity;
       int nz;
       real zsam;
       real zz;
       real z1;
       real zf;

       # cosmo
       real omegam=0.3;
       real omegal=0.7;
       real h0=70;

       real c=2.99792458e5;
       real cmpermpc=3.08568e24;

       nz=0;
       while (nz*0.02<z){
             nz+=1;
       }
       nz-=1
       //nz = floor(z/0.02);
       zsam = z/nz;
       zz = zsam*0.5;
       for (i in 1:nz){
            z1 = zz+1.0;
            zf = (z1**2/sqrt(omegam*z1**3+omegal));
            zf *= zsam*c*n0*cmpermpc/h0;
            opacity = absori_opacity_free_T_n0(energybounds, t4, n0, zz,
                      abund, ionizing_spec, xil, atomicnumber,
                      energy_base, sigma, ion);
            opacity_int += xsec*zf;
            zz+=zsam;
            }

       return opacity_int;
}
