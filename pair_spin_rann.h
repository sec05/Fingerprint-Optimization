

#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <map>
#include <dirent.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include <sys/resource.h>
//#include <bits/stdc++.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include "utils.h"
#define MAXLINE 4096
#define SHORTLINE 128
#define NEIGHMASK 0x3FFFFFFF
#define FLERR __FILE__,__LINE__

#ifndef CALIBRATION_H_
#define CALIBRATION_H_

namespace LAMMPS_NS{

namespace RANN { 
	class Activation;
	class Fingerprint;
	class State;
}

class PairRANN{
public:
	PairRANN(char *);
	~PairRANN();
	void setup();

	//global parameters read from file
	char *algorithm;
	char *potential_input_file;
	char *dump_directory;
	bool doforces;
	double tolerance;
	double regularizer;
	bool doregularizer;
	char *log_file;
	char *potential_output_file;
	int potential_output_freq;
	int max_epochs;
	bool overwritepotentials;
	int debug_level1_freq;
	int debug_level2_freq;
	int debug_level3_freq;
	int debug_level4_freq;
	int debug_level5_freq;
	int debug_level5_spin_freq;
	int debug_level6_freq;
	bool adaptive_regularizer;
	double lambda_initial;
	double lambda_increase;
	double lambda_reduce;
	int seed;
	double validation;
	bool normalizeinput;
	int targettype;
	double inum_weight;

	//global variables calculated internally
	bool is_lammps = false;
	char *lmp = nullptr;
	int nsims;
	int nsets;
	int betalen;
	int jlen1;
	int *betalen_v;
	int *betalen_f;
	int natoms;
	int natomsr;
	int natomsv;
	int fmax;
	int fnmax;
	int *r;//simulations included in training
	int *v;//simulations held back for validation
	int nsimr,nsimv;
	int *Xset;
	char **dumpfilenames;
	double **normalshift;
	double **normalgain;
	bool ***weightdefined;
	bool ***biasdefined;
	bool **dimensiondefined;
	bool ***bundle_inputdefined;
	bool ***bundle_outputdefined;
	double energy_fitv_best;
	int nelements;                // # of elements (distinct from LAMMPS atom types since multiple atom types can be mapped to one element)
	int nelementsp;				// nelements+1
	char **elements;              // names of elements
	char **elementsp;				// names of elements with "all" appended as the last "element"
	double *mass;                 // mass of each element
	double cutmax;				// max radial distance for neighbor lists
	int *map;                     // mapping from atom types to elements
	int *fingerprintcount;		// static variable used in initialization
	int *fingerprintlength;       // # of input neurons defined by fingerprints of each element.
	int *fingerprintperelement;   // # of fingerprints for each element
	int *stateequationperelement;
	int *stateequationcount;
	bool doscreen;//screening is calculated if any defined fingerprint uses it
	bool allscreen;
	bool dospin;
	int res;//Resolution of function tables for cubic interpolation.
	double *screening_min;
	double *screening_max;
	int memguess;
	bool *freezebeta;
	int speciesnumberr;
	int speciesnumberv;
	bool freeenergy;
	double hbar;

	struct NNarchitecture{
	  int layers;
	  int *dimensions;//vector of length layers with entries for neurons per layer
	  int *activations;//unused
	  int maxlayer;//longest layer (for memory allocation)
	  int sumlayers;
	  int *startI;
	  bool bundle;
	  int *bundles;
	  int **bundleinputsize;
	  int **bundleoutputsize;
	  bool **identitybundle;
	  int ***bundleinput;
	  int ***bundleoutput;
	  double ***bundleW;
	  double ***bundleB;
	  bool ***freezeW;
	  bool ***freezeB;
	};
	NNarchitecture *net;//array of networks, 1 for each element.

	struct Simulation{
		bool forces;
		bool spins;
		int *id;
		double **x;
		double **f;
		double **s;
		double box[3][3];
		double origin[3];
		double **features;
		double **dfx;
		double **dfy;
		double **dfz;
		double **dsx;
		double **dsy;
		double **dsz;
		int *ilist,*numneigh,**firstneigh,*type,inum,gnum;
		double energy;
		double energy_weight;
		double force_weight;
		int startI;
		char *filename;
		int timestep;
		bool spinspirals;
		double spinvec[3];
		double spinaxis[3];
		double **force;
		double **fm;
		double state_e;
		double *state_ea;
		double *total_ea;
		double time;
		int uniquespecies;
		int *speciesmap;
		int speciesoffset;
		int atomoffset;
		int *speciescount;
		double temp;
	};
	Simulation *sims;

	//read potential file:
	void read_file(char *);
  void read_atom_types(std::vector<std::string>, char *, int);
  void read_fpe(
      std::vector<std::string>, std::vector<std::string>, char *,
      int);    //fingerprints per element. Count total fingerprints defined for each 1st element in element combinations
  void read_fingerprints(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_fingerprint_constants(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_network_layers(std::vector<std::string>, std::vector<std::string>, char *,
                           int);    //include input and output layer (hidden layers + 2)
  void read_layer_size(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_weight(std::vector<std::string>, std::vector<std::string>, FILE *, char *,
                   int *);    //weights should be formatted as properly shaped matrices
  void read_bias(std::vector<std::string>, std::vector<std::string>, FILE *, char *,
                 int *);    //biases should be formatted as properly shaped vectors
  void read_activation_functions(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_screening(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_mass(const std::vector<std::string> &, const std::vector<std::string> &, const char *,
                 int);
  void read_eospe(std::vector<std::string>, std::vector<std::string>, char *,int);    
  void read_eos(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_eos_constants(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundles(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_input(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_output(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_bundle_id(std::vector<std::string>, std::vector<std::string>, char *, int);
  void read_parameters(std::vector<std::string>, std::vector<std::string>, FILE *, char *, int*,char *);
	bool check_potential();

	//process_data
	void read_dump_files();
	void create_neighbor_lists();
	void screen(double*,double*,double*,double*,double*,double*,double*,bool*,int,int,double*,double*,double*,int *,int);
	void cull_neighbor_list(double *,double *,double *,int *,int *,int *,int,int,double);
	void screen_neighbor_list(double *,double *,double *,int *,int *,int *,int,int,bool*,double*,double*,double*,double*,double*,double*,double*);
	void compute_fingerprints();
	void separate_validation();
	int count_unique_species(int*,int);

	//handle network
	void create_random_weights(int,int,int,int,int);
	void create_random_biases(int,int,int,int);
	void create_identity_wb(int,int,int,int,int);
	void copy_network(NNarchitecture*,NNarchitecture*);
	void unnormalize_net(NNarchitecture*);

	//utility and misc
	void allocate(const std::vector<std::string> &);//called after reading element list, but before reading the rest of the potential
	bool check_parameters();	
	void update_stack_size();
	int factorial(int);
	void errorf(const std::string&, int,const char *);
	void errorf(char *, int,const char *);
	void errorf(const char *);
	std::vector<std::string> tokenmaker(std::string,std::string);
	int count_words(char *);
	int count_words(char *,char *);
	void qrsolve(double *,int,int,double*,double *);
	void chsolve(double *,int,double*,double *);

	//create styles
  	RANN::Fingerprint *create_fingerprint(const char *);
  	RANN::Activation *create_activation(const char *);
	RANN::State *create_state(const char *);

 protected:
  RANN::Activation ****activation;
  RANN::Fingerprint ***fingerprints;
  RANN::State ***state;
};



}
#endif /* CALIBRATION_H_ */
