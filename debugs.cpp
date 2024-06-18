#include "pair_spin_rann.h"
#include "rann_fingerprint.h"

using namespace LAMMPS_NS;

//Calculate numerical derivatives of features wrt to neighbor list and compare with analytical derivatives
void PairRANN::write_debug_level6(double *fit_err,double *val_err) {
	printf("starting debug level 6\n");
	printf("Very slow. Do NOT run this on more than a handful of atoms!\n");
	double diff = 1e-5;//numerical derivative resolution
	FILE *dumps1[nsims];
	FILE *dumps2[nsims];
	FILE *current1;
	FILE *current2;
	char *debugnames1[nsims];
	char *debugnames2[nsims];
	int check = mkdir("DEBUG",0777);
	for (int i=0;i<nsims;i++){
		debugnames1[i] = new char [strlen(sims[i].filename)+30];
		sprintf(debugnames1[i],"DEBUG/%s.features.debug6.%d.csv",sims[i].filename,sims[i].timestep);
		debugnames2[i] = new char [strlen(sims[i].filename)+30];
		sprintf(debugnames2[i],"DEBUG/%s.screening.debug6.%d.csv",sims[i].filename,sims[i].timestep);
	}
	int fmax=0;
	for (int itype=0;itype<nelementsp;itype++){
		if (net[itype].layers==0)continue;
		if (net[itype].dimensions[0]>fmax)fmax=net[itype].dimensions[0];
	}
	#pragma omp parallel
	{
	int i,ii,itype,f,jnum,len,j,nn,s,k,v;
	#pragma omp for schedule(guided)
	for (nn=0;nn<nsims;nn++){
		int nmax = 0;
		for (j=0;j<sims[nn].inum;j++){
			if (nmax<sims[nn].numneigh[j])nmax=sims[nn].numneigh[j];
		}
		dumps1[nn]=fopen(debugnames1[nn],"w");
		dumps2[nn]=fopen(debugnames2[nn],"w");
		int j;
		current1 = dumps1[nn];
		current2 = dumps2[nn];
		fprintf(current1,"sim,atom,neighbor,neighbor_type,type,neighbor_id");
		fprintf(current2,"sim,atom,neighbor,neighbor_type,type,neighbor_id,Sik");
		for (f=0;f<fmax;f++){
			fprintf(current1,",features%d,dfxn%d,dfxa%d,dfyn%d,dfya%d,dfzn%d,dfza%d",f,f,f,f,f,f,f);
		}
		fprintf(current1,"\n");
		for (f=0;f<nmax;f++){
			fprintf(current2,",dSikxn%d,dSikxa%d,dSikyn%d,dSikya%d,dSikzn%d,dSikza%d",f,f,f,f,f,f);
		}
		fprintf(current2,"\n");
		for (ii=0;ii<sims[nn].inum;ii++){
			i = sims[nn].ilist[ii];
			itype = map[sims[nn].type[i]];
			f = net[itype].dimensions[0];
			jnum = sims[nn].numneigh[i];
			double xn[jnum];
			double yn[jnum];
			double zn[jnum];
			int tn[jnum];
			int jl[jnum];
			cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,cutmax);
			double features1[jnum][f][3][2];
			double dfeatures1[jnum][f][3];
			double screenik[jnum][3][2];
			double dscreenik[jnum][3];
			double screenijk[jnum][jnum][3][2];
			double dscreenijk[jnum][jnum][3];
			double features [f];
			double dfeaturesx[f*jnum];
			double dfeaturesy[f*jnum];
			double dfeaturesz[f*jnum];
			int jnum1 = jnum;
			for (int jj=0;jj<jnum1;jj++){
				int j1 = sims[nn].firstneigh[i][jj];
				for (v=0;v<3;v++){		
					for (s=-1;s<2;s=s+2){
						//printf("%d %d %d %d %d\n",nn,ii,jj,v,s);
						if (jj<jnum-1){
							if (v==0)sims[nn].x[j1][0]-=s*diff;
							else if (v==1)sims[nn].x[j1][1]-=s*diff;
							else if (v==2)sims[nn].x[j1][2]-=s*diff;
						}
						else {
							if (v==0)sims[nn].x[i][0]-=s*diff;
							else if (v==1)sims[nn].x[i][1]-=s*diff;
							else if (v==2)sims[nn].x[i][2]-=s*diff;
						}
						jnum = sims[nn].numneigh[i];
						cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,cutmax);
						for (j=0;j<f;j++){
							features[j]=0;
						}
						for (j=0;j<f*jnum;j++){
							dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0.0;
						}
						//screening is calculated once for all atoms if any fingerprint uses it.
						double Sik[jnum];
						double dSikx[jnum];
						double dSiky[jnum];
						double dSikz[jnum];
						//TO D0: add check against stack size
						double dSijkx[jnum*jnum];
						double dSijky[jnum*jnum];
						double dSijkz[jnum*jnum];
						bool Bij[jnum];
						double sx[jnum*f];
						double sy[jnum*f];
						double sz[jnum*f];
						double sxx[jnum*f];
						double sxy[jnum*f];
						double sxz[jnum*f];
						double syy[jnum*f];
						double syz[jnum*f];
						double szz[jnum*f];
						for (j=0;j<f*jnum;j++){
							sx[j]=sy[j]=sz[j]=0;
							sxx[j]=sxy[j]=sxz[j]=syy[j]=syz[j]=szz[j]=0;
						}
						if (doscreen){
							screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
						}
						//do fingerprints for atom type
						len = fingerprintperelement[itype];
						for (j=0;j<len;j++) {
							if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
						}
						itype = nelements;
						//do fingerprints for type "all"
						len = fingerprintperelement[itype];
						for (j=0;j<len;j++) {
							if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
							else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
						}
						for (j=0;j<f;j++){
							features1[jj][j][v][(s+1)>>1]=features[j];
						}
						screenik[jj][v][(s+1)>>1]=Sik[jj];
						for (j=0;j<jnum-1;j++){
							screenijk[jj][j][v][(s+1)>>1]=Sik[j];
						}
						itype = map[sims[nn].type[i]];
						if (jj<jnum-1){
							if (v==0)sims[nn].x[j1][0]+=s*diff;
							else if (v==1)sims[nn].x[j1][1]+=s*diff;
							else if (v==2)sims[nn].x[j1][2]+=s*diff;
						}
						else {
							if (v==0)sims[nn].x[i][0]+=s*diff;
							else if (v==1)sims[nn].x[i][1]+=s*diff;
							else if (v==2)sims[nn].x[i][2]+=s*diff;
						}
					}
					for (j=0;j<f;j++){
						dfeatures1[jj][j][v]=(features1[jj][j][v][1]-features1[jj][j][v][0])/diff/2;
					}
					dscreenik[jj][v] = (screenik[jj][v][1]-screenik[jj][v][0])/diff/2;
					for (j=0;j<jnum-1;j++){
						dscreenijk[jj][j][v] = (screenijk[jj][j][v][1]-screenijk[jj][j][v][0])/diff/2;
					}
				}
				jnum = sims[nn].numneigh[i];
				cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,cutmax);
				double features [f];
				double dfeaturesx[f*jnum];
				double dfeaturesy[f*jnum];
				double dfeaturesz[f*jnum];
				for (j=0;j<f;j++){
					features[j]=0;
				}
				for (j=0;j<f*jnum;j++){
					dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
				}
				//screening is calculated once for all atoms if any fingerprint uses it.
				double Sik[jnum];
				double dSikx[jnum];
				double dSiky[jnum];
				double dSikz[jnum];
				//TO D0: add check against stack size
				double dSijkx[jnum*jnum];
				double dSijky[jnum*jnum];
				double dSijkz[jnum*jnum];
				bool Bij[jnum];
				double sx[jnum*f];
				double sy[jnum*f];
				double sz[jnum*f];
				double sxx[jnum*f];
				double sxy[jnum*f];
				double sxz[jnum*f];
				double syy[jnum*f];
				double syz[jnum*f];
				double szz[jnum*f];
				for (j=0;j<f*jnum;j++){
					sx[j]=sy[j]=sz[j]=0;
					sxx[j]=sxy[j]=sxz[j]=syy[j]=syz[j]=szz[j]=0;
				}
				if (doscreen){
					screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);
				}
				if (jj<jnum-1){
					fprintf(current2,"%d, %d, %d, %d, %d, %d",nn,ii,jj,tn[jj],itype,jl[jj]);
					fprintf(current2,", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e",Sik[jj],dscreenik[jj][0],dSikx[jj]*Sik[jj],dscreenik[jj][1],dSiky[jj]*Sik[jj],dscreenik[jj][2],dSikz[jj]*Sik[jj]);
					for (j=0;j<jnum-1;j++){
						if (j==jj)fprintf(current2,", 0, 0, 0, 0, 0, 0");
						else fprintf(current2,", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e",dscreenijk[jj][j][0],dSijkx[j*(jnum-1)+jj]*Sik[j],dscreenijk[jj][j][1],dSijky[j*(jnum-1)+jj]*Sik[j],dscreenijk[jj][j][2],dSijkz[j*(jnum-1)+jj]*Sik[j]);
					}
					fprintf(current2,"\n");
				}
				//do fingerprints for atom type
				len = fingerprintperelement[itype];
				for (j=0;j<len;j++) {
					if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				}
				itype = nelements;
				//do fingerprints for type "all"
				len = fingerprintperelement[itype];
				for (j=0;j<len;j++) {
					if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				}
				itype = map[sims[nn].type[i]];
				fprintf(current1,"%d, %d, %d, %d, %d, %d",nn,ii,jj,tn[jj],itype,jl[jj]);
				for (j=0;j<f;j++){
					fprintf(current1,", %.10e, %.10e, %.10e, %.10e, %.10e, %.10e, %.10e",features[j],dfeatures1[jj][j][0],dfeaturesx[jj*f+j],dfeatures1[jj][j][1],dfeaturesy[jj*f+j],dfeatures1[jj][j][2],dfeaturesz[jj*f+j]);
				}
				fprintf(current1,"\n");
			}		
		}
		fclose(dumps1[i]);
		fclose(dumps2[i]);
	}
	}
}