#include "omp.h"
#include "pair_spin_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"


using namespace LAMMPS_NS;

PairRANN::PairRANN(char *potential_file){
	cutmax = 0.0;
	nelementsp = -1;
	nelements = -1;
	net = NULL;
	fingerprintlength = NULL;
	mass = NULL;
	betalen = 0;
	doregularizer = false;
	normalizeinput = true;
	fingerprints = NULL;
	max_epochs = 1e7;
	regularizer = 0.0;
	res = 10000;
	fingerprintcount = 0;
	stateequationcount = 0;
	elementsp = NULL;
	elements = NULL;
	activation = NULL;
	state = NULL;
	tolerance = 1e-6;
	sims = NULL;
	doscreen = false;
	allscreen = true;
	dospin = false;
	map = NULL;//check this
	natoms = 0;
	nsims = 0;
	doforces = false;
	fingerprintperelement = NULL;
	stateequationperelement = NULL;
	validation = 0.0;
	potential_output_freq = 100;
	algorithm = new char [SHORTLINE];
	potential_input_file = new char [strlen(potential_file)+1];
	dump_directory = new char [SHORTLINE];
	log_file = new char [SHORTLINE];
	potential_output_file = new char [SHORTLINE];
	strncpy(this->potential_input_file,potential_file,strlen(potential_file)+1);
	char temp1[] = ".\0";
	char temp2[] = "calibration.log\0";
	char temp3[] = "potential_output.nn\0";
	strncpy(dump_directory,temp1,strlen(temp1)+1);
	strncpy(log_file,temp2,strlen(temp2)+1);
	strncpy(potential_output_file,temp3,strlen(temp3)+1);
	strncpy(algorithm,"LMch",strlen("LMch")+1);
	overwritepotentials = false;
	debug_level1_freq = 10;
	debug_level2_freq = 0;
	debug_level3_freq = 0;
	debug_level4_freq = 0;
	debug_level5_freq = 0;
	debug_level5_spin_freq = 0;
	debug_level6_freq = 0;
	adaptive_regularizer = false;
	seed = time(0);
	lambda_initial = 1000;
	lambda_increase = 10;
	lambda_reduce = 0.3;
	targettype = 1;
	inum_weight = 1.0;
	freeenergy = 0;
	hbar = 4.135667403e-3/6.283185307179586476925286766559;
}

PairRANN::~PairRANN(){
	//clear memory
	delete [] algorithm;
	delete [] potential_input_file;
	delete [] dump_directory;
	delete [] log_file;
	delete [] potential_output_file;
	delete [] r;
	delete [] v;
	delete [] Xset;
	delete [] mass;
	for (int i=0;i<nsims;i++){
		for (int j=0;j<sims[i].inum;j++){
//			delete [] sims[i].x[j];
			if (doforces)delete [] sims[i].f[j];
			if (sims[i].spins)delete [] sims[i].s[j];
			delete [] sims[i].firstneigh[j];
			delete [] sims[i].features[j];
			if (doforces)delete [] sims[i].dfx[j];
			if (doforces)delete [] sims[i].dfy[j];
			if (doforces)delete [] sims[i].dfz[j];
		}
//		delete [] sims[i].x;
		if (doforces)delete [] sims[i].f;
		if (sims[i].spins)delete [] sims[i].s;
		if (doforces)delete [] sims[i].dfx;
		if (doforces)delete [] sims[i].dfy;
		if (doforces)delete [] sims[i].dfz;
		if (targettype>1)delete [] sims[i].total_ea;
		delete [] sims[i].firstneigh;
		delete [] sims[i].id;
		delete [] sims[i].features;
		delete [] sims[i].ilist;
		delete [] sims[i].numneigh;
		delete [] sims[i].type;
	}
	delete [] sims;
	for (int i=0;i<nelements;i++){delete [] elements[i];}
	delete [] elements;
	for (int i=0;i<nelementsp;i++){delete [] elementsp[i];}
	delete [] elementsp;
	for (int i=0;i<=nelements;i++){
		if (net[i].layers>0){
			for (int j=0;j<net[i].layers-1;j++){
				delete [] net[i].bundleinputsize[j];
				delete [] net[i].bundleoutputsize[j];
				for (int k=0;k<net[i].dimensions[j+1];k++){
					delete activation[i][j][k];
				}
				for (int k=0;k<net[i].bundles[j];k++){
					delete [] net[i].bundleinput[j][k];
					delete [] net[i].bundleoutput[j][k];
					delete [] net[i].bundleW[j][k];
					delete [] net[i].bundleB[j][k];
					delete [] net[i].freezeW[j][k];
					delete [] net[i].freezeB[j][k];
				}
				delete [] activation[i][j];
				delete [] net[i].bundleinput[j];
				delete [] net[i].bundleoutput[j];
				delete [] net[i].bundleW[j];
				delete [] net[i].bundleB[j];
				delete [] net[i].freezeW[j];
				delete [] net[i].freezeB[j];
			}
			delete [] activation[i];
			delete [] net[i].dimensions;
			delete [] net[i].startI;
			delete [] net[i].bundleinput;
			delete [] net[i].bundleoutput;
			delete [] net[i].bundleinputsize;
			delete [] net[i].bundleoutputsize;
			delete [] net[i].bundleW;
			delete [] net[i].bundleB;
			delete [] net[i].freezeW;
			delete [] net[i].freezeB;
			delete [] net[i].bundles;
		}
	}
	delete [] net;
	delete [] map;
	for (int i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			for (int j=0;j<fingerprintperelement[i];j++){
				delete fingerprints[i][j];
			}
			delete [] fingerprints[i];
		}
		if (stateequationperelement[i]>0){
			for (int j=0;j<stateequationperelement[i];j++){
				delete state[i][j];
			}
			delete [] state[i];
		}
	}
	delete [] fingerprints;
	delete [] activation;
	delete [] state;
	delete [] fingerprintcount;
	delete [] fingerprintperelement;
	delete [] fingerprintlength;
	delete [] stateequationcount;
	delete [] stateequationperelement;
	delete [] freezebeta;
}

void PairRANN::setup(){

	int nthreads=1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();

	std::cout << std::endl;
	std::cout << "# Number Threads     : " << nthreads << std::endl;

	double start_time = omp_get_wtime();

	read_file(potential_input_file);
	check_parameters();
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<fingerprintperelement[i];j++){
			  fingerprints[i][j]->allocate();
		}
		for (int j=0;j<stateequationperelement[i];j++){
			state[i][j]->allocate();
		}
	}
	read_dump_files();
	create_neighbor_lists();
	compute_fingerprints();
	separate_validation();

	double end_time = omp_get_wtime();
	double time = (end_time-start_time);
	printf("finished setup(): %f seconds\n",time);
}


void PairRANN::read_parameters(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum,char *linetemp){
	if (line[1]=="algorithm"){
		if (line[1].size()>SHORTLINE){
			delete [] algorithm;
			algorithm = new char[line[1].size()+1];
		}
		strncpy(algorithm,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="dumpdirectory"){
		if (line1[0].size()>SHORTLINE){
			delete [] dump_directory;
			dump_directory = new char[line1[0].size()+1];
		}
		strncpy(dump_directory,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="doforces"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		doforces = (temp>0);
	}
	else if (line[1]=="normalizeinput"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		normalizeinput = (temp>0);
	}
	else if (line[1]=="tolerance"){
		tolerance = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="regularizer"){
		regularizer = strtod(line1[0].c_str(),NULL);
		if (regularizer!=0.0){doregularizer = true;}
	}
	else if (line[1]=="logfile"){
		delete [] log_file;
		log_file = new char[strlen(linetemp)];
		strncpy(log_file,linetemp,strlen(linetemp)-1);
	}
	else if (line[1]=="potentialoutputfreq"){
		potential_output_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="potentialoutputfile"){
		delete [] potential_output_file;
		potential_output_file = new char[strlen(linetemp)];
		strncpy(potential_output_file,linetemp,strlen(linetemp)-1);
	}
	else if (line[1]=="maxepochs"){
		max_epochs = strtol(line1[0].c_str(),NULL,10);
	}
	// else if (line[1]=="dimsreserved"){
	// 	int i;
	// 	for (i=0;i<nelements;i++){
	// 		if (strcmp(words[2],elements[i])==0){
	// 			if (net[i].layers==0)errorf("networklayers for each atom type must be defined before the corresponding layer sizes.");
	// 			int j = strtol(words[3],NULL,10);
	// 			net[i].dimensionsr[j]= strtol(line1,NULL,10);
	// 			return;
	// 		}
	// 	}
	// 	errorf("dimsreserved element not found in atom types");
	// }
	else if (line[1]=="freezeW"){
		int i,j,k,b,l,ins,ops;
		char **words1,*ptr;
		char linetemp [MAXLINE];
		int nwords = line.size();
		if (nwords == 4){b=0;}
		else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
		for (l=0;l<nelements;l++){
			if (line[2]==elements[l]){
				if (net[l].layers==0)errorf("networklayers must be defined before weights.");
				i=strtol(line[3].c_str(),NULL,10);
				if (i>=net[l].layers || i<0)errorf("invalid weight layer");
				if (dimensiondefined[l][i]==false || dimensiondefined[l][i+1]==false) errorf("network layer sizes must be defined before corresponding weight");
				if (bundle_inputdefined[l][i][b]==false && b!=0) errorf("bundle inputs must be defined before weights");
				if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before weights");
				if (net[l].identitybundle[i][b]) errorf("cannot define weights for an identity bundle");
				if (bundle_inputdefined[l][i][b]==false){ins = net[l].dimensions[i];}
				else {ins = net[l].bundleinputsize[i][b];}
				if (bundle_outputdefined[l][i][b]==false){ops = net[l].dimensions[i+1];}
				else {ops = net[l].bundleoutputsize[i][b];}
				net[l].freezeW[i][b] = new bool [ins*ops];
				nwords = line1.size();
				if (nwords != ins)errorf("invalid weights per line");
				for (k=0;k<ins;k++){
					net[l].freezeW[i][b][k] = strtol(line1[k].c_str(),NULL,10);
				}
				for (j=1;j<ops;j++){
					ptr = fgets(linetemp,MAXLINE,fp);
					(linenum)++;
					line1 = tokenmaker(linetemp,": ,\t_\n");
					if (ptr==NULL)errorf("unexpected end of potential file!");
					nwords = line1.size();
					if (nwords != ins)errorf("invalid weights per line");
					for (k=0;k<ins;k++){
						net[l].freezeW[i][b][j*ins+k] = strtol(line1[k].c_str(),NULL,10);
					}
				}
				delete [] words1;
				return;
			}
		}
		errorf("weight element not found in atom types");
	}
	else if (line[1]=="freezeB"){
		int i,j,l,b,ops;
		char *ptr;
		int nwords = line.size();
		char linetemp[MAXLINE];
		if (nwords == 4){b=0;}
		else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
		for (l=0;l<nelements;l++){
			if (line[2]==elements[l]){
				if (net[l].layers==0)errorf("networklayers must be defined before biases.");
				i=strtol(line[3].c_str(),NULL,10);
				if (i>=net[l].layers || i<0)errorf("invalid bias layer");
				if (dimensiondefined[l][i]==false) errorf("network layer sizes must be defined before corresponding bias");
				if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before bias");
				if (net[l].identitybundle[i][b]) errorf("cannot define bias for an identity bundle");
				if (bundle_outputdefined[l][i][b]==false){ops=net[l].dimensions[i+1];}
				else {ops = net[l].bundleoutputsize[i][b];}
				net[l].freezeB[i][b] = new bool [ops];
				net[l].freezeB[i][b][0] = strtol(line1[0].c_str(),NULL,10);
				for (j=1;j<ops;j++){
					ptr = fgets(linetemp,MAXLINE,fp);
					if (ptr==NULL)errorf("unexpected end of potential file!");
					line1 = tokenmaker(linetemp," ,\t:_\n");
					net[l].freezeB[i][b][j] = strtol(line1[0].c_str(),NULL,10);
				}
				return;
			}
		}
		errorf("bias element not found in atom types");
	}
	else if (line[1]=="validation"){
		validation = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="overwritepotentials") {
		int temp = strtol(line1[0].c_str(),NULL,10);
		overwritepotentials = (temp>0);
	}
	else if (line[1]=="debug1freq") {
		debug_level1_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug2freq") {
		debug_level2_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug3freq") {
		debug_level3_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug4freq") {
		debug_level4_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug5freq") {
		debug_level5_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug5spinfreq") {
		debug_level5_spin_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug6freq") {
		debug_level6_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="adaptiveregularizer") {
		double temp = strtod(line1[0].c_str(),NULL);
		adaptive_regularizer = (temp>0);
		doregularizer = true;
	}
	else if (line[1]=="lambdainitial"){
		lambda_initial = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdaincrease"){
		lambda_increase = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdareduce"){
		lambda_reduce = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="seed"){
		seed = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="targettype"){
		targettype = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="inumweight"){
		inum_weight = strtod(line1[0].c_str(),NULL);
	}
	else {
		char str[MAXLINE];
		//sprintf(str,"unrecognized keyword in parameter file: %s\n",line[1]);
		errorf(filename,*linenum,str);
	}
}

void PairRANN::create_random_weights(int rows,int columns,int itype,int layer,int bundle){
	net[itype].bundleW[layer][bundle] = new double [rows*columns];
	net[itype].freezeW[layer][bundle] = new bool [rows*columns];
	double r;
	for (int i=0;i<rows;i++){
		for (int j=0;j<columns;j++){
			r = (double)rand()/RAND_MAX*2-1;//flat distribution from -1 to 1
			net[itype].bundleW[layer][bundle][i*columns+j] = r;
			net[itype].freezeW[layer][bundle][i*columns+j] = 0;
		}
	}
	weightdefined[itype][layer][bundle]=true;
}

void PairRANN::create_random_biases(int rows,int itype, int layer,int bundle){
	net[itype].bundleB[layer][bundle] = new double [rows];
	net[itype].freezeB[layer][bundle] = new bool [rows];
	double r;
	for (int i=0;i<rows;i++){
		r = (double) rand()/RAND_MAX*2-1;
		net[itype].bundleB[layer][bundle][i] = r;
		net[itype].freezeB[layer][bundle][i] = 0;
	}
	biasdefined[itype][layer][bundle]=true;
}

void PairRANN::allocate(const std::vector<std::string> &elementwords)
{
	int i,n;
	cutmax = 0;
	nelementsp=nelements+1;
	//initialize arrays
	elements = new char *[nelements];
	elementsp = new char *[nelementsp];//elements + 'all'
	map = new int[nelementsp];
	mass = new double[nelements];
	net = new NNarchitecture[nelementsp];
	for (i=0;i<nelementsp;i++){net[i].layers=0;}
	betalen_v = new int[nelementsp];
	betalen_f = new int[nelementsp];
	screening_min = new double [nelements*nelements*nelements];
	screening_max = new double [nelements*nelements*nelements];
	for (i=0;i<nelements;i++){
		for (int j =0;j<nelements;j++){
			for (int k=0;k<nelements;k++){
				screening_min[i*nelements*nelements+j*nelements+k] = 0.8;//default values. Custom values may be read from potential file later.
				screening_max[i*nelements*nelements+j*nelements+k] = 2.8;//default values. Custom values may be read from potential file later.
			}
		}
	}
	weightdefined = new bool**[nelementsp];
	biasdefined = new bool **[nelementsp];
	dimensiondefined = new bool*[nelements];
	bundle_inputdefined = new bool**[nelements];
	bundle_outputdefined = new bool**[nelements];
	activation = new RANN::Activation***[nelementsp];
	fingerprints = new RANN::Fingerprint**[nelementsp];
	state = new RANN::State**[nelementsp];
	fingerprintlength = new int[nelementsp];
	fingerprintperelement = new int [nelementsp];
	fingerprintcount = new int[nelementsp];
	stateequationperelement = new int [nelementsp];
	stateequationcount = new int [nelementsp];
	for (i=0;i<=nelements;i++){
		n = elementwords[i].size();
		fingerprintlength[i]=0;
		fingerprintperelement[i] = -1;
		fingerprintcount[i] = 0;
		stateequationperelement[i] = 0;
		stateequationcount[i] = 0;
		map[i] = i;
		if (i<nelements){
			mass[i]=-1.0;
			elements[i]= utils::strdup(elementwords[i]);
		}
		elementsp[i]= utils::strdup(elementwords[i]);
	}

}


void PairRANN::update_stack_size(){
	//TO DO: fix. Still getting stack overflow from underestimating memory needs.
	//get very rough guess of memory usage
	int jlen = nsims;
	if (doregularizer){
		jlen+=betalen-1;
	}
	if (doforces){
		jlen+=natoms*3;
	}
	//neighborlist memory use:
	memguess = 0;
	for (int i=0;i<nelementsp;i++){
		memguess+=8*net[i].dimensions[0]*20*3;
	}
	memguess+=8*20*12;
	memguess+=8*20*20*3;
	//separate validation memory use:
	memguess+=nsims*8*2;
	//levenburg marquardt ch memory use:
	memguess+=8*jlen*betalen*2;
	memguess+=8*betalen*betalen;
	memguess+=8*jlen*4;
	memguess+=8*betalen*4;
	//chsolve memory use:
	memguess+=8*betalen*betalen;
	//generous buffer:
	memguess *= 16;
	const rlim_t kStackSize = memguess;
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur += kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

bool PairRANN::check_parameters(){
	int itype,layer,bundle,rows,columns,r,c,count;
	if (strcmp(algorithm,"LMqr")!=0 && strcmp(algorithm,"LMch")!=0 && strcmp(algorithm,"CG")!=0 && strcmp(algorithm,"LMsearch")!=0 && strcmp(algorithm,"bfgs")!=0)errorf(FLERR,"Unrecognized algorithm. Must be CG, LMch or LMqr\n");//add others later maybe
	if (tolerance==0.0)errorf(FLERR,"tolerance not correctly initialized\n");
	if (tolerance<0.0 || max_epochs < 0 || regularizer < 0.0 || potential_output_freq < 0)errorf(FLERR,"detected parameter with negative value which must be positive.\n");
	if (targettype>3 || targettype<1)errorf(FLERR,"targettype must be 1, 2, or 3.");
	srand(seed);
	count=0;
	//populate vector of frozen parameters
	betalen = 0;
	for (itype=0;itype<nelementsp;itype++){
		for (layer=0;layer<net[itype].layers-1;layer++){
			for (bundle=0;bundle<net[itype].bundles[layer];bundle++){
				if (net[itype].identitybundle[layer][bundle]){continue;}
				rows = net[itype].bundleoutputsize[layer][bundle];
				columns = net[itype].bundleinputsize[layer][bundle];
				betalen += rows*columns+rows;
			}
		}
	}
	freezebeta = new bool[betalen];
	for (itype=0;itype<nelementsp;itype++){
		for (layer=0;layer<net[itype].layers-1;layer++){
			for (bundle=0;bundle<net[itype].bundles[layer];bundle++){
				if (net[itype].identitybundle[layer][bundle]){continue;}
				rows = net[itype].bundleoutputsize[layer][bundle];
				columns = net[itype].bundleinputsize[layer][bundle];
				for (r=0;r<rows;r++){
					for (c=0;c<columns;c++){
						if (net[itype].freezeW[layer][bundle][r*columns+c]){
							freezebeta[count] = 1;
							count++;
						}
						else {
							freezebeta[count] = 0;
							count++;
						}
					}
					if (net[itype].freezeB[layer][bundle][r]){
						freezebeta[count] = 1;
						count++;
					}
					else {
						freezebeta[count] = 0;
						count++;
					}
				}
			}
		}
		betalen_v[itype]=count;
	}
	betalen = count;//update betalen to skip frozen parameters

	return false;//everything looks good
}

//part of setup. Do not optimize:
void PairRANN::read_dump_files(){
	DIR *folder;
//	char str[MAXLINE];
	struct dirent *entry;
	int file = 0;
	char line[MAXLINE],*ptr;
	char **words;
	int nwords,nwords1,sets;
	folder = opendir(dump_directory);

	if(folder == NULL)
	{
		errorf("unable to open dump directory");
	}
	std::cout<<"reading dump files\n";
	int nsims = 0;
	int nsets = 0;
	//count files
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		nsets++;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	this->nsets = nsets;
	Xset=new int[nsets];
	dumpfilenames = new char*[nsets];
	int count=0;
	//count snapshots per file
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		dumpfilenames[count] = new char[strlen(entry->d_name)+10];
		strcpy(dumpfilenames[count],entry->d_name);
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		ptr = fgets(line,MAXLINE,fid);
		nwords = 0;
		words = new char* [strlen(line)];
		words[nwords++] = strtok(line," ,\t\n");
		while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
		nwords--;
		if (nwords!=5 && nwords != 11 && nwords != 6 && nwords != 12){errorf(entry->d_name,2,"dumpfile must contain 2nd line with timestep, energy, energy_weight, force_weight, snapshots\n");}
		sets = strtol(words[4],NULL,10);
		delete [] words;
		nsims+=sets;
		Xset[count++]=sets;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	sims = new Simulation[nsims];
	this->nsims = nsims;
	sims[0].startI=0;
	//read dump files
	while((entry=readdir(folder))){
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		printf("\t%s\n",entry->d_name);
		if (!fid){continue;}
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		while (ptr!=NULL){
			if (strstr(line,"ITEM: TIMESTEP")==NULL)errorf("invalid dump file line 1");
			ptr = fgets(line,MAXLINE,fid);//timestep
			nwords = 0;
			char *words1[strlen(line)];
			words1[nwords++] = strtok(line," ,\t");
			while ((words1[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			if (nwords!=5 && nwords != 11 && nwords != 6 && nwords != 12)errorf("error: dump file line 2 must contain 5 entries: timestep, energy, energy_weight, force_weight, snapshots");
			int timestep = strtol(words1[0],NULL,10);
			sims[file].filename = new char [strlen(entry->d_name)+10];
			sims[file].timestep = timestep;
			strcpy(sims[file].filename,entry->d_name);
			sims[file].energy = strtod(words1[1],NULL);
			sims[file].energy_weight = strtod(words1[2],NULL);
			sims[file].force_weight = strtod(words1[3],NULL);
			sims[file].spinvec[0] = 0;
			sims[file].spinvec[1] = 0;
			sims[file].spinvec[2] = 0;
			sims[file].spinaxis[0] = 0;
			sims[file].spinaxis[1] = 0;
			sims[file].spinaxis[2] = 0;
			sims[file].temp = 0;
			if (nwords==6){
				freeenergy = 1;
				sims[file].temp = strtod(words1[5],NULL);
			}
			if (nwords==11){
				sims[file].spinspirals = true;
				double spinvec[3],spinaxis[3];
				spinvec[0] = strtod(words1[5],NULL);
				spinvec[1] = strtod(words1[6],NULL);
				spinvec[2] = strtod(words1[7],NULL);
				spinaxis[0] = strtod(words1[8],NULL);
				spinaxis[1] = strtod(words1[9],NULL);
				spinaxis[2] = strtod(words1[10],NULL);
				double norm = spinaxis[0]*spinaxis[0]+spinaxis[1]*spinaxis[1]+spinaxis[2]*spinaxis[2];
				if (norm<1e-14){errorf("spinaxis cannot be zero\n");}
				spinaxis[0]=spinaxis[0]/sqrt(norm);
				spinaxis[1]=spinaxis[1]/sqrt(norm);
				spinaxis[2]=spinaxis[2]/sqrt(norm);
				sims[file].spinvec[0] = spinvec[0];
				sims[file].spinvec[1] = spinvec[1];
				sims[file].spinvec[2] = spinvec[2];
				sims[file].spinaxis[0] = spinaxis[0];
				sims[file].spinaxis[1] = spinaxis[1];
				sims[file].spinaxis[2] = spinaxis[2];
			}
			if (nwords==12){
				freeenergy = 1;
				sims[file].temp = strtod(words1[11],NULL);
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: NUMBER OF ATOMS
			if (strstr(line,"ITEM: NUMBER OF ATOMS")==NULL)errorf("invalid dump file line 3");
			ptr = fgets(line,MAXLINE,fid);//natoms
			int natoms = strtol(line,NULL,10);
			//printf("%d %d %f\n",file,timestep,sims[file].energy/natoms);
			if (file>0){sims[file].startI=sims[file-1].startI+natoms*3;}
			this->natoms+=natoms;
			if (targettype==1)sims[file].energy_weight /=pow(natoms,inum_weight);
			//sims[file].force_weight /=natoms;
			ptr = fgets(line,MAXLINE,fid);//ITEM: BOX BOUNDS xy xz yz pp pp pp
			if (strstr(line,"ITEM: BOX BOUNDS")==NULL)errorf("invalid dump file line 5");
			double box[3][3];
			double origin[3];
			bool cols[12];
			for (int i= 0;i<11;i++){
				cols[i]=false;
			}
			box[0][1] = box[0][2] = box[1][2] = 0.0;
			for (int i = 0;i<3;i++){
				ptr = fgets(line,MAXLINE,fid);//box line
				char *words[4];
				nwords = 0;
				words[nwords++] = strtok(line," ,\t\n");
				while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
				nwords--;
				if (nwords!=3 && nwords!=2){errorf("invalid dump box definition");}
				origin[i] = strtod(words[0],NULL);
				box[i][i] = strtod(words[1],NULL);
				if (nwords==3){
					if (i==0){
						box[0][1]=strtod(words[2],NULL);
						if (box[0][1]>0){box[0][0]-=box[0][1];}
						else origin[0] -= box[0][1];
					}
					else if (i==1){
						box[0][2]=strtod(words[2],NULL);
						if (box[0][2]>0){box[0][0]-=box[0][2];}
						else origin[0] -= box[0][2];
					}
					else{
						box[1][2]=strtod(words[2],NULL);
						if (box[1][2]>0)box[1][1]-=box[1][2];
						else origin[1] -=box[1][2];
					}
				}
			}
			for (int i=0;i<3;i++)box[i][i]-=origin[i];
			box[1][0]=box[2][0]=box[2][1]=0.0;
			ptr = fgets(line,MAXLINE,fid);//ITEM: ATOMS id type x y z c_energy fx fy fz sx sy sz
			nwords = 0;
			char *words[count_words(line)+1];
			words[nwords++] = strtok(line," ,\t\n");
			while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			int colid = -1;
			int columnmap[11];
			for (int i=0;i<nwords-2;i++){columnmap[i]=-1;}
			for (int i=2;i<nwords;i++){
				if (strcmp(words[i],"type")==0){colid = 0;}
				else if (strcmp(words[i],"x")==0){colid=1;}
				else if (strcmp(words[i],"y")==0){colid=2;}
				else if (strcmp(words[i],"z")==0){colid=3;}
				else if (strcmp(words[i],"fx")==0){colid=4;}
				else if (strcmp(words[i],"fy")==0){colid=5;}
				else if (strcmp(words[i],"fz")==0){colid=6;}
				else if (strcmp(words[i],"sx")==0){colid=7;}
				else if (strcmp(words[i],"sy")==0){colid=8;}
				else if (strcmp(words[i],"sz")==0){colid=9;}
				else if (strcmp(words[i],"c_eng")==0){colid=10;}
				else {continue;}
				cols[colid] = true;
				if (colid!=-1){columnmap[colid]=i-2;}
			}
			for (int i=0;i<4;i++){
				if (!cols[i]){errorf("dump file must include type, x, y, and z data columns (other recognized keywords are fx, fy, fz, sx, sy, sz)");}
			}
			bool doforce = false;
			bool dospin = false;
			sims[file].inum = natoms;
			sims[file].ilist = new int [natoms];
			sims[file].type = new int [natoms];
			sims[file].x= new double *[natoms];
			for (int i=0;i<3;i++){
				for (int j=0;j<3;j++)sims[file].box[i][j]=box[i][j];
				sims[file].origin[i]=origin[i];
			}
			for (int i=0;i<natoms;i++){
				sims[file].x[i]=new double [3];
			}
			//if force calibration is on
			if (doforces){
				sims[file].f = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].f[i] = new double [3];
				}
			}
			//if forces are given in dump file
			if (cols[4] && cols[5] && cols[6] && doforces){
				doforce = true;
				sims[file].forces=doforce;
			}
			//if per-atom energies will be used
			if (targettype>1)sims[file].total_ea = new double [natoms];
			//if spin vectors are provided
			if (cols[7] && cols[8] && cols[9]){
				dospin = true;
				sims[file].s = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].s[i] = new double [3];
				}
			}
			else if (this->dospin){
				errorf(FLERR,"spin vectors must be defined for all input simulations when magnetic fingerprints are used");
			}
			if (!cols[10] && targettype>1){
				errorf(FLERR,"per-atom energies must be specified using keyword \"c_eng\" in all input dump files when per-atom or per-species training is enabled");
			}
			for (int i=0;i<natoms;i++){
				ptr = fgets(line,MAXLINE,fid);
				char *words2[count_words(line)+1];
				nwords1 = 0;
				words2[nwords1++] = strtok(line," ,\t");
				while ((words2[nwords1++] = strtok(NULL," ,\t"))) continue;
				nwords1--;
				if (nwords1!=nwords-2){errorf("incorrect number of data columns in dump file.");}
				sims[file].ilist[i]=i;//ignore any id mapping in the dump file, just id them based on line number.
				sims[file].type[i]=strtol(words2[columnmap[0]],NULL,10)-1;//lammps type counting starts at 1 instead of 0
				sims[file].x[i][0]=strtod(words2[columnmap[1]],NULL);
				sims[file].x[i][1]=strtod(words2[columnmap[2]],NULL);
				sims[file].x[i][2]=strtod(words2[columnmap[3]],NULL);
				//sims[file].energy[i]=strtod(words[columnmap[4]],NULL);
				if (doforce){
					sims[file].f[i][0]=strtod(words2[columnmap[4]],NULL);
					sims[file].f[i][1]=strtod(words2[columnmap[5]],NULL);
					sims[file].f[i][2]=strtod(words2[columnmap[6]],NULL);
				}
				//if force calibration is on, but forces are not given in file, assume they are zero.
				else if (doforces){
					sims[file].f[i][0]=0.0;
					sims[file].f[i][1]=0.0;
					sims[file].f[i][2]=0.0;
				}
				if (dospin){
					sims[file].s[i][0]=strtod(words2[columnmap[7]],NULL);
					sims[file].s[i][1]=strtod(words2[columnmap[8]],NULL);
					sims[file].s[i][2]=strtod(words2[columnmap[9]],NULL);
					double sm = sims[file].s[i][0]*sims[file].s[i][0]+sims[file].s[i][1]*sims[file].s[i][1]+sims[file].s[i][2]*sims[file].s[i][2];
					sims[file].s[i][0]/=sqrt(sm);
					sims[file].s[i][1]/=sqrt(sm);
					sims[file].s[i][2]/=sqrt(sm);
				}
				sims[file].spins = dospin;
				if (targettype>1){
					sims[file].total_ea[i] = strtod(words2[columnmap[10]],NULL);
				}
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
			file++;
			if (file>nsims){errorf("Too many dump files found. Nsims is incorrect.\n");}
		}
		fclose(fid);
	}
	closedir(folder);
	sprintf(line,"imported %d atoms, %d simulations\n",natoms,nsims);
	std::cout<<line;
}

int PairRANN::count_unique_species(int *s,int nsims){
	int nn,n1,j,count1,count2,count3;
	count1=0;
	count3=0;
	for (n1=0;n1<nsims;n1++){
		nn = s[n1];
		sims[nn].speciescount = new int[nelements];
		sims[nn].speciesoffset = count1;
		sims[nn].atomoffset = count3;
		sims[nn].speciesmap = new int[nelements];
		bool tp[nelements];
		for (j=0;j<nelements;j++){
			tp[j]=false;
			sims[nn].speciescount[j] = 0;
		}
		for (j=0;j<sims[nn].inum;j++){
			tp[sims[nn].type[j]]=true;
			sims[nn].speciescount[sims[nn].type[j]]++;
		}
		count2 = 0;
		for (j=0;j<nelements;j++){
			if (tp[j]==true) {
				sims[nn].speciesmap[j]=count2;
				count1++;
				count2++;
			}
		}
		count3 += sims[nn].inum;
		sims[nn].uniquespecies=count2;	
	}
	return count1;
}

//part of setup. Do not optimize:
void PairRANN::create_neighbor_lists(){
	//brute force search technique rather than tree search because we only do it once and most simulations are small.
	//I did optimize for low memory footprint by only adding ghost neighbors
	//within cutoff distance of the box
	int i,ix,iy,iz,j,k;
//	char str[MAXLINE];
	double buffer = 0.01;//over-generous compensation for roundoff error
	std::cout<<"building neighbor lists\n";
	for (i=0;i<nsims;i++){
		double box[3][3];
		for (ix=0;ix<3;ix++){
			for (iy=0;iy<3;iy++)box[ix][iy]=sims[i].box[ix][iy];
		}
		double *origin = sims[i].origin;
		int natoms = sims[i].inum;
		int xb = floor(cutmax/box[0][0]+1);
		int yb = floor(cutmax/box[1][1]+1);
		int zb = floor(cutmax/box[2][2]+1);
		int buffsize = natoms*(xb*2+1)*(yb*2+1)*(zb*2+1);
		double x[buffsize][3];
		int type[buffsize];
		int id[buffsize];
		double spins[buffsize][3];
		int count = 0;
		//force all atoms to be inside the box:
		double xtemp[3];
		double xp[3];
		double boxt[9];
		for (j=0;j<3;j++){
			for (k=0;k<3;k++){
				boxt[j*3+k]=box[j][k];
			}
		}
		for (j=0;j<natoms;j++){
			for (k=0;k<3;k++){
				xp[k] = sims[i].x[j][k]-origin[k];
			}
			qrsolve(boxt,3,3,xp,xtemp);//convert coordinates from Cartesian to box basis (uses qrsolve for matrix inversion)
			for (k=0;k<3;k++){
				xtemp[k]-=floor(xtemp[k]);//if atom is outside box find periodic replica in box
			}
			for (k=0;k<3;k++){
				sims[i].x[j][k] = 0.0;
				for (int l=0;l<3;l++){
					sims[i].x[j][k]+=box[k][l]*xtemp[l];//convert back to Cartesian
				}
				sims[i].x[j][k]+=origin[k];
			}
		}

		//calculate box face normal directions and plane intersections
		double xpx,xpy,xpz,ypx,ypy,ypz,zpx,zpy,zpz;
		zpx = 0;zpy=0;zpz =1;
		double ym,xm;
		ym = sqrt(box[1][2]*box[1][2]+box[2][2]*box[2][2]);
		xm = sqrt(box[1][1]*box[2][2]*box[1][1]*box[2][2]+box[0][1]*box[0][1]*box[2][2]*box[2][2]+(box[0][1]*box[1][2]-box[0][2]*box[1][1])*(box[0][1]*box[1][2]-box[0][2]*box[1][1]));
		//unit vectors normal to box faces:
		ypx = 0;
		ypy = box[2][2]/ym;
		ypz = -box[1][2]/ym;
		xpx = box[1][1]*box[2][2]/xm;
		xpy = -box[0][1]*box[2][2]/xm;
		xpz = (box[0][1]*box[1][2]-box[0][2]*box[1][1])/xm;
		double fxn,fxp,fyn,fyp,fzn,fzp;
		//minimum distances from origin to planes aligned with box faces:
		fxn = origin[0]*xpx+origin[1]*xpy+origin[2]*xpz;
		fyn = origin[0]*ypx+origin[1]*ypy+origin[2]*ypz;
		fzn = origin[0]*zpx+origin[1]*zpy+origin[2]*zpz;
		fxp = (origin[0]+box[0][0])*xpx+(origin[1]+box[1][0])*xpy+(origin[2]+box[2][0])*xpz;
		fyp = (origin[0]+box[0][1])*ypx+(origin[1]+box[1][1])*ypy+(origin[2]+box[2][1])*ypz;
		fzp = (origin[0]+box[0][2])*zpx+(origin[1]+box[1][2])*zpy+(origin[2]+box[2][2])*zpz;
		//fill buffered atom list
		double px,py,pz;
		double xe,ye,ze;
		double theta,sx,sy,sz;
		for (j=0;j<natoms;j++){
			x[count][0] = sims[i].x[j][0];
			x[count][1] = sims[i].x[j][1];
			x[count][2] = sims[i].x[j][2];
			type[count] = sims[i].type[j];
			if (sims[i].spins){
				spins[count][0] = sims[i].s[j][0];
				spins[count][1] = sims[i].s[j][1];
				spins[count][2] = sims[i].s[j][2];
			}
			id[count] = j;
			count++;
		}

		//add ghost atoms outside periodic boundaries:
		for (ix=-xb;ix<=xb;ix++){
			for (iy=-yb;iy<=yb;iy++){
				for (iz=-zb;iz<=zb;iz++){
					if (ix==0 && iy == 0 && iz == 0)continue;
					for (j=0;j<natoms;j++){
						xe = ix*box[0][0]+iy*box[0][1]+iz*box[0][2]+sims[i].x[j][0];
						ye = iy*box[1][1]+iz*box[1][2]+sims[i].x[j][1];
						ze = iz*box[2][2]+sims[i].x[j][2];
						px = xe*xpx+ye*xpy+ze*xpz;
						py = xe*ypx+ye*ypy+ze*ypz;
						pz = xe*zpx+ye*zpy+ze*zpz;
						//include atoms if their distance from the box face is less than cutmax
						if (px>cutmax+fxp+buffer || px<fxn-cutmax-buffer){continue;}
						if (py>cutmax+fyp+buffer || py<fyn-cutmax-buffer){continue;}
						if (pz>cutmax+fzp+buffer || pz<fzn-cutmax-buffer){continue;}
						x[count][0] = xe;
						x[count][1] = ye;
						x[count][2] = ze;
						type[count] = sims[i].type[j];
						id[count] = j;
						if (sims[i].spinspirals && sims[i].spins){
							// sims[i].s[j][0]=1;
							// sims[i].s[j][2]=0;
							// spins[j][0]=1;
							// spins[j][2]=0;
							theta = sims[i].spinvec[0]*(ix*box[0][0]+iy*box[0][1]+iz*box[0][2]) + sims[i].spinvec[1]*(iy*box[1][1]+iz*box[1][2]) + sims[i].spinvec[2]*iz*box[2][2];
							double sxi = sims[i].s[j][0];
							double syi = sims[i].s[j][1];
							double szi = sims[i].s[j][2];
							double ax = sims[i].spinaxis[0];
							double ay = sims[i].spinaxis[1];
							double az = sims[i].spinaxis[2];
							ax = 0;//REMOVE
							az = 1;//REMOVE
							sx = sxi*(ax*ax*(1-cos(theta))+cos(theta))+syi*(ax*ay*(1-cos(theta))-az*sin(theta))+szi*(ax*az*(1-cos(theta))+ay*sin(theta));
							sy = sxi*(ax*ay*(1-cos(theta))+az*sin(theta))+syi*(ay*ay*(1-cos(theta))+cos(theta))+szi*(-ax*sin(theta)+ay*az*(1-cos(theta)));
							sz = sxi*(ax*az*(1-cos(theta))-ay*sin(theta))+syi*(ax*sin(theta)+ay*az*(1-cos(theta)))+szi*(az*az*(1-cos(theta))+cos(theta));
							//sx = ax*(ax*sxi+ay*syi+az*szi)+(ay*szi-az*syi)*sin(theta)+(-ay*(ax*syi-ay*sxi)+az*(-ax*szi+az*sxi))*cos(theta);
							//sy = ay*(ax*sxi+ay*syi+az*szi)+(-ax*szi+az*sxi)*sin(theta)+(ax*(ax*syi-ay*sxi)-az*(ay*szi-az*syi))*cos(theta);
							//sz = az*(ax*sxi+ay*syi+az*szi)+(ax*syi-ay*sxi)*sin(theta)+(-ax*(-ax*szi+az*sxi)-ay*(ay*szi-az*syi))*cos(theta);
							spins[count][0]=sx;
							spins[count][1]=sy;
							spins[count][2]=sz;
						}
						else if (sims[i].spins) {
							spins[count][0]=sims[i].s[j][0];
							spins[count][1]=sims[i].s[j][1];
							spins[count][2]=sims[i].s[j][2];
						}
						count++;
						if (count>buffsize){errorf("neighbor overflow!\n");}
					}
				}
			}
		}

		//update stored lists
		buffsize = count;
		for (j=0;j<natoms;j++){
			delete [] sims[i].x[j];
		}
		delete [] sims[i].x;
		delete [] sims[i].type;
		delete [] sims[i].ilist;
		if (sims[i].spins){
			for (j=0;j<natoms;j++){
				delete [] sims[i].s[j];
			}
			delete [] sims[i].s;
			sims[i].s = new double *[buffsize];
		}
		sims[i].type = new int [buffsize];
		sims[i].x = new double *[buffsize];
		sims[i].id = new int [buffsize];
		sims[i].ilist = new int [buffsize];

		for (j=0;j<buffsize;j++){
			sims[i].x[j] = new double [3];
			for (k=0;k<3;k++){
				sims[i].x[j][k] = x[j][k];
			}
			sims[i].type[j] = type[j];
			sims[i].id[j] = id[j];
			sims[i].ilist[j] = j;
			if (sims[i].spins){
				sims[i].s[j] = new double [3];
				for (k=0;k<3;k++){
					sims[i].s[j][k]=spins[j][k];
				}
			}
		}
		sims[i].inum = natoms;
		sims[i].gnum = buffsize-natoms;
		sims[i].numneigh = new int[natoms];
		sims[i].firstneigh = new int*[natoms];
		//do double count, slow, but enables getting the exact size of the neighbor list before filling it.
		for (j=0;j<natoms;j++){
			sims[i].numneigh[j]=0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].numneigh[j]++;
				}
			}
			sims[i].firstneigh[j] = new int[sims[i].numneigh[j]];
			count = 0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].firstneigh[j][count] = k;
					count++;
				}
			}
		}
	}
}

//part of setup. Do not optimize:
//TO DO: fix stack size problem
void PairRANN::compute_fingerprints(){
	std::cout<<"computing fingerprints\n";
	int nn,j,ii,f,i,itype,jnum;
	for (nn=0;nn<nsims;nn++){
		sims[nn].features = new double *[sims[nn].inum];
		sims[nn].state_e = 0;
		if (doforces){
			sims[nn].dfx = new double *[sims[nn].inum];
			sims[nn].dfy = new double *[sims[nn].inum];
			sims[nn].dfz = new double *[sims[nn].inum];
			if (dospin){
				sims[nn].dsx = new double *[sims[nn].inum];
				sims[nn].dsy = new double *[sims[nn].inum];
				sims[nn].dsz = new double *[sims[nn].inum];
			}
		}
		sims[nn].force = new double*[sims[nn].inum+sims[nn].gnum];
		sims[nn].fm = new double*[sims[nn].inum+sims[nn].gnum];
		  for (j=0;j<sims[nn].inum+sims[nn].gnum;j++){
			  sims[nn].force[j]=new double[3];
			  sims[nn].fm[j]=new double[3];
			  sims[nn].force[j][0]=0;
			  sims[nn].force[j][1]=0;
			  sims[nn].force[j][2]=0;
			  sims[nn].fm[j][0]=0;
			  sims[nn].fm[j][1]=0;
			  sims[nn].fm[j][2]=0;
		  }
			for (ii=0;ii<sims[nn].inum;ii++){
				i = sims[nn].ilist[ii];
			  	itype = map[sims[nn].type[i]];
				if (net[itype].layers==0){errorf(FLERR,"atom type found without corresponding network defined");}
			    f = net[itype].dimensions[0];
				jnum = sims[nn].numneigh[i];
				sims[nn].features[ii] = new double [f];
				if (doforces){
				  sims[nn].dfx[ii] = new double[f*jnum];
				  sims[nn].dfy[ii] = new double[f*jnum];
				  sims[nn].dfz[ii] = new double[f*jnum];
				  if (dospin){
					  sims[nn].dsx[ii] = new double[f*jnum];
					  sims[nn].dsy[ii] = new double[f*jnum];
					  sims[nn].dsz[ii] = new double[f*jnum];
				  }
			   }
		  }
		}
		#pragma omp parallel
		{
		int i,ii,itype,f,jnum,len,j,nn;
		double **force,**fm;
		#pragma omp for schedule(guided)
		for (nn=0;nn<nsims;nn++){
		  clock_t start = clock();
		
		  double start_time = omp_get_wtime();
		  force = sims[nn].force;
		  fm = sims[nn].fm;
		  if (debug_level2_freq>0){
			  sims[nn].state_ea = new double [sims[nn].inum];
		  }
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
			  //TO D0: stack overflow often happens here from stack limit too low.
			  double dSijkx[jnum*jnum];
			  double dSijky[jnum*jnum];
			  double dSijkz[jnum*jnum];
			  //TO D0: stack overflow often happens here from stack limit too low.
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
					screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);//jnum is neighlist + self term, hence jnum-1 in function inputs
			  }
			  if (allscreen){
				  screen_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,Bij,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz);
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
			  //copy features from stack to heap
			  for (j=0;j<f;j++){
				  sims[nn].features[ii][j] = features[j];
			  }
			  if (doforces){
				  for (j=0;j<f*jnum;j++){
					  sims[nn].dfx[ii][j]=dfeaturesx[j];
					  sims[nn].dfy[ii][j]=dfeaturesy[j];
					  sims[nn].dfz[ii][j]=dfeaturesz[j];
				  }
				  if (dospin){
					  for (j=0;j<f*jnum;j++){
						  sims[nn].dsx[ii][j] = sx[j];
						  sims[nn].dsy[ii][j] = sy[j];
						  sims[nn].dsz[ii][j] = sz[j];
					  }
				  }
			  }
			  double e=0.0;
			  itype = map[sims[nn].type[i]];
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  itype = nelements;
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  sims[nn].energy-=e;
			  sims[nn].state_e+=e;
			  if (debug_level2_freq>0){sims[nn].state_ea[ii]=e;}
			  if (targettype>1){sims[nn].total_ea[ii]-=e;}
		  }
		  clock_t end = clock();
		  sims[nn].time = (double)(end-start)/ CLOCKS_PER_SEC;
	}
	}
}


void PairRANN::unnormalize_net(NNarchitecture *net_out){
	int i,j,k;
	double temp;
	copy_network(net,net_out);
	for (i=0;i<nelementsp;i++){
		if (net[i].layers>0){
			for (int i1=0;i1<net[i].bundles[0];i1++){
				for (j=0;j<net[i].bundleoutputsize[0][i1];j++){
					temp = 0.0;
					for (k=0;k<net[i].bundleinputsize[0][i1];k++){
						if (normalgain[i][k]>0){
							net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]/=normalgain[i][net[i].bundleinput[0][i1][k]];
							temp+=net_out[i].bundleW[0][i1][j*net[i].bundleinputsize[0][i1]+k]*normalshift[i][net[i].bundleinput[0][i1][k]];
						}
					}
					net_out[i].bundleB[0][i1][j]-=temp;
				}
			}
		}
	}
}


void PairRANN::separate_validation(){
	int n1,n2,i,vnum,len,startI,endI,j,t,k;
	char str[MAXLINE];
	int Iv[nsims];
	int Ir[nsims];
	bool w;
	n1=n2=0;
	sprintf(str,"finishing setup\n");
	std::cout<<str;
	for (i=0;i<nsims;i++)Iv[i]=-1;
	for (i=0;i<nsets;i++){
		startI=0;
		for (j=0;j<i;j++)startI+=Xset[j];
		endI = startI+Xset[i];
		len = Xset[i];
		// vnum = rand();
		// if (vnum<floor(RAND_MAX*validation)){
		// 	vnum = 1;
		// }
		// else{
		// 	vnum = 0;
		// }
		vnum = 0;// if Xset has only 1 entry, do not include it in validation ever. (Code above puts it randomly in validation or fit).
		vnum+=floor(len*validation);
		while (vnum>0){
			w = true;
			t = floor(rand() % len)+startI;
			for (j=0;j<n1;j++){
				if (t==Iv[j]){
					w = false;
					break;
				}
			}
			if (w){
				Iv[n1]=t;
				vnum--;
				n1++;
			}
		}
		for (j=startI;j<endI;j++){
			w = true;
			for (k=0;k<n1;k++){
				if (j==Iv[k]){
					w = false;
					break;
				}
			}
			if (w){
				Ir[n2]=j;
				n2++;
			}
		}
	}
	nsimr = n2;
	nsimv = n1;
	r = new int [n2];
	v = new int [n1];
	natomsr = 0;
	natomsv = 0;
	for (i=0;i<n1;i++){
		v[i]=Iv[i];
		natomsv += sims[v[i]].inum;
	}
	for (i=0;i<n2;i++){
		r[i]=Ir[i];
		natomsr += sims[r[i]].inum;
	}
	sprintf(str,"assigning %d simulations (%d atoms) for validation, %d simulations (%d atoms) for fitting\n",nsimv,natomsv,nsimr,natomsr);
	std::cout<<str;
}

void PairRANN::copy_network(NNarchitecture *net_old,NNarchitecture *net_new){
	int i,j,k;
	for (i=0;i<nelementsp;i++){
		net_new[i].layers = net_old[i].layers;
		if (net_new[i].layers>0){
			net_new[i].maxlayer = net_old[i].maxlayer;
			net_new[i].sumlayers=net_old[i].sumlayers;
			net_new[i].dimensions = new int [net_new[i].layers];
			net_new[i].startI = new int [net_new[i].layers];
			net_new[i].bundleW = new double**[net_new[i].layers-1];
			net_new[i].bundleB = new double**[net_new[i].layers-1];
			net_new[i].freezeW = new bool**[net_new[i].layers-1];
			net_new[i].freezeB = new bool**[net_new[i].layers-1];
			net_new[i].bundleinputsize = new int*[net_new[i].layers-1];
			net_new[i].bundleoutputsize = new int*[net_new[i].layers-1];
			net_new[i].bundleinput = new int**[net_new[i].layers-1];
			net_new[i].bundleoutput = new int**[net_new[i].layers-1];
			net_new[i].bundles = new int [net_new[i].layers-1];
			net_new[i].identitybundle = new bool *[net_new[i].layers-1];
			for (j=0;j<net_old[i].layers;j++){
				net_new[i].dimensions[j]=net_old[i].dimensions[j];
				net_new[i].startI[j]=net_old[i].startI[j];
				if (j==net_old[i].layers-1)continue;
				net_new[i].bundles[j]=net_old[i].bundles[j];
				net_new[i].bundleW[j] = new double*[net_new[i].bundles[j]];
				net_new[i].bundleB[j] = new double*[net_new[i].bundles[j]];
				net_new[i].freezeW[j] = new bool*[net_new[i].bundles[j]];
				net_new[i].freezeB[j] = new bool*[net_new[i].bundles[j]];
				net_new[i].identitybundle[j] = new bool[net_new[i].bundles[j]];
				net_new[i].bundleinputsize[j] = new int[net_new[i].bundles[j]];
				net_new[i].bundleoutputsize[j] = new int [net_new[i].bundles[j]];
				net_new[i].bundleinput[j] = new int*[net_new[i].bundles[j]];
				net_new[i].bundleoutput[j] = new int*[net_new[i].bundles[j]];
				for (int i1=0;i1<net_old[i].bundles[j];i1++){
					net_new[i].identitybundle[j][i1]=net_old[i].identitybundle[j][i1];
					net_new[i].bundleinputsize[j][i1]=net_old[i].bundleinputsize[j][i1];
					net_new[i].bundleoutputsize[j][i1]=net_old[i].bundleoutputsize[j][i1];
					net_new[i].bundleinput[j][i1] = new int[net_new[i].bundleinputsize[j][i1]];
					net_new[i].bundleoutput[j][i1] = new int[net_new[i].bundleoutputsize[j][i1]];
					net_new[i].bundleW[j][i1] = new double[net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1]];
					net_new[i].bundleB[j][i1] = new double[net_new[i].bundleoutputsize[j][i1]];
					net_new[i].freezeW[j][i1] = new bool[net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1]];
					net_new[i].freezeB[j][i1] = new bool[net_new[i].bundleoutputsize[j][i1]];
					for (int k=0;k<net_new[i].bundleinputsize[j][i1]*net_new[i].bundleoutputsize[j][i1];k++){
						net_new[i].bundleW[j][i1][k] = net_old[i].bundleW[j][i1][k];
						net_new[i].freezeW[j][i1][k] = net_old[i].freezeW[j][i1][k];
					}
					for (int k=0;k<net_new[i].bundleinputsize[j][i1];k++){
						net_new[i].bundleinput[j][i1][k]=net_old[i].bundleinput[j][i1][k];
					}
					for (int k=0;k<net_new[i].bundleoutputsize[j][i1];k++){
						net_new[i].bundleoutput[j][i1][k]=net_old[i].bundleoutput[j][i1][k];
						net_new[i].bundleB[j][i1][k]=net_old[i].bundleB[j][i1][k];
						net_new[i].freezeB[j][i1][k]=net_old[i].freezeB[j][i1][k];
					}
				}
			}
		}
	}
}





void PairRANN::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,double cutmax){
	int *jlist,j,count,jj,*type,jtype;
	double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double **x = sims[sn].x;
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	type = sims[sn].type;
	jlist = sims[sn].firstneigh[i];
	count = 0;
	for (jj=0;jj<jnum[0];jj++){
		j = jlist[jj];
		j &= NEIGHMASK;
		jtype = map[type[j]];
		delx = xtmp - x[j][0];
		dely = ytmp - x[j][1];
		delz = ztmp - x[j][2];
		rsq = delx*delx + dely*dely + delz*delz;
		if (rsq>cutmax*cutmax){
			continue;
		}
		xn[count]=delx;
		yn[count]=dely;
		zn[count]=delz;
		tn[count]=jtype;
		//jl[count]=sims[sn].id[j];
		jl[count]=j;
		//jl is currently only used to calculate spin dot products.
		//j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
		//lammps code uses id instead of j because spin spirals are not supported.
		count++;
	}
	jnum[0]=count+1;
}

void PairRANN::screen_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,bool *Bij,double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz){
	double xnc[jnum[0]],ync[jnum[0]],znc[jnum[0]];
	double Sikc[jnum[0]];
	double dSikxc[jnum[0]];
	double dSikyc[jnum[0]];
	double dSikzc[jnum[0]];
	double dSijkxc[jnum[0]][jnum[0]];
	double dSijkyc[jnum[0]][jnum[0]];
	double dSijkzc[jnum[0]][jnum[0]];
	int jj,kk,count,count1,tnc[jnum[0]],jlc[jnum[0]];
	count = 0;
	for (jj=0;jj<jnum[0]-1;jj++){
		if (Bij[jj]){
			count1 = 0;
			xnc[count]=xn[jj];
			ync[count]=yn[jj];
			znc[count]=zn[jj];
			tnc[count]=tn[jj];
			jlc[count]=jl[jj];
			Sikc[count]=Sik[jj];
			dSikxc[count]=dSikx[jj];
			dSikyc[count]=dSiky[jj];
			dSikzc[count]=dSikz[jj];
			for (kk=0;kk<jnum[0]-1;kk++){
				if (Bij[kk]){
					dSijkxc[count][count1] = dSijkx[jj*(jnum[0]-1)+kk];
					dSijkyc[count][count1] = dSijky[jj*(jnum[0]-1)+kk];
					dSijkzc[count][count1] = dSijkz[jj*(jnum[0]-1)+kk];
					count1++;
				}
			}
			count++;
		}
	}
	jnum[0]=count+1;
	for (jj=0;jj<count;jj++){
		xn[jj]=xnc[jj];
		yn[jj]=ync[jj];
		zn[jj]=znc[jj];
		tn[jj]=tnc[jj];
		jl[jj]=jlc[jj];
		Bij[jj] = true;
		Sik[jj]=Sikc[jj];
		dSikx[jj]=dSikxc[jj];
		dSiky[jj]=dSikyc[jj];
		dSikz[jj]=dSikzc[jj];
		for (kk=0;kk<count;kk++){
			dSijkx[jj*count+kk] = dSijkxc[jj][kk];
			dSijky[jj*count+kk] = dSijkyc[jj][kk];
			dSijkz[jj*count+kk] = dSijkzc[jj][kk];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//replaced with Cholesky solution for greater speed for finding solve step. Still used to process input data.
void PairRANN::qrsolve(double *A,int m,int n,double *b, double *x_){
	double QR_[m*n];
//	char str[MAXLINE];
	double Rdiag[n];
	int i=0, j=0, k=0;
	int j_off, k_off;
	double nrm;
    // loop to copy QR from A.
	for (k=0;k<n;k++){
		k_off = k*m;
		for (i=0;i<m;i++){
			QR_[k_off+i]=A[i*n+k];
		}
	}
    for (k = 0; k < n; k++) {
       // Compute 2-norm of k-th column.
       nrm = 0.0;
       k_off = k*m;
       for (i = k; i < m; i++) {
			nrm += QR_[k_off+i]*QR_[k_off+i];
       }
       if (nrm==0.0){
    	   errorf(FLERR,"Jacobian is rank deficient!\n");
       }
       nrm = sqrt(nrm);
	   // Form k-th Householder vector.
	   if (QR_[k_off+k] < 0) {
		 nrm = -nrm;
 	   }
	   for (i = k; i < m; i++) {
		 QR_[k_off+i] /= nrm;
	   }
	   QR_[k_off+k] += 1.0;

	   // Apply transformation to remaining columns.
	   for (j = k+1; j < n; j++) {
		 double s = 0.0;
		 j_off = j*m;
		 for (i = k; i < m; i++) {
			s += QR_[k_off+i]*QR_[j_off+i];
		 }
		 s = -s/QR_[k_off+k];
		 for (i = k; i < m; i++) {
			QR_[j_off+i] += s*QR_[k_off+i];
		 }
	   }
       Rdiag[k] = -nrm;
    }
    //loop to find least squares
    for (int j=0;j<m;j++){
    	x_[j] = b[j];
    }
    // Compute Y = transpose(Q)*b
	for (int k = 0; k < n; k++)
	{
		k_off = k*m;
		double s = 0.0;
		for (int i = k; i < m; i++)
		{
		   s += QR_[k_off+i]*x_[i];
		}
		s = -s/QR_[k_off+k];
		for (int i = k; i < m; i++)
		{
		   x_[i] += s*QR_[k_off+i];
		}
	}
	// Solve R*X = Y;
	for (int k = n-1; k >= 0; k--)
	{
		k_off = k*m;
		x_[k] /= Rdiag[k];
		for (int i = 0; i < k; i++) {
		   x_[i] -= x_[k]*QR_[k_off+i];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
void PairRANN::chsolve(double *A,int n,double *b, double *x){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	int	nthreads=omp_get_num_threads();

	double L_[n*n]; // was L_[n][n]
	int i,j,k;
	int iXn, jXn, kXn;
	double d, s;

	// initialize L
	for (k=0;k<n*n;k++){
		L_[k]=0.0;
	}

	// Cholesky-Crout decomposition
	#pragma omp parallel default(none) shared (A,L_,n,s)
	{
	for (int j = 0; j <n; j++) {
		int jXn = j*n;
		s = 0.0;
		// #pragma omp for schedule(static) reduction(+:s)
		for (int k = 0; k < j; k++) {
			s += L_[jXn + k] * L_[jXn + k];
		}
		#pragma omp barrier
		double d = A[jXn+j] - s;
		#pragma omp single
		{
		if (d>0){
			L_[jXn + j] = sqrt(d);
		}
		}
		//// #pragma omp parallel for schedule(static) default(none) shared (A,L_,n,j,jXn)
		////#pragma omp barrier
		#pragma omp for schedule(static)
		for (int i = j+1; i <n; i++) {
			int iXn = i * n;
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += L_[iXn + k] * L_[jXn + k];
			}
			L_[iXn + j] =  (A[iXn + j] - sum) / L_[jXn + j];
		}
	}
	}
	// Solve L*
	// Forward substitution to solve L*y = b;
	// #pragma omp parallel default(none) shared (x,b,L_,n,s) private(i)
	// #pragma omp parallel
	{
	for (int k = 0; k < n; k++)
	{
		int kXn = k*n;
		s = 0.0;
		// #pragma omp parallel for default(none) reduction(+:s) schedule(static) shared (x,L_,kXn,k) private(i) if (nthreads>k)
		// #pragma omp for reduction(+:s) schedule(static)
		for (i = 0; i < k; i++) {
			s += x[i]*L_[kXn+i];
		}
		// #pragma omp single
		x[k] = (b[k] - s) / L_[kXn+k];
	}
	}
	// Backward substitution to solve L'*X = Y; omp does not work
	for (int k = n-1; k >= 0; k--)
	{
		double s = 0.0;
		for (int i = k+1; i < n; i++) {
			s += x[i]*L_[i*n+k];
		}
		x[k] = (x[k] - s)/L_[k*n+k];
	}

	//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	//printf(" - chsolve(): %f ms\n",time);

	return;
}

void PairRANN::screen(double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz, bool *Bij, int ii,int sid,double *xn,double *yn,double *zn,int *tn,int jnum)
{
	//#pragma omp parallel
	{
	//see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
	int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
	double Sijk,Cijk,Cn,Cd,Dij,Dik,Djk,C,dfc,dC,**x;
	PairRANN::Simulation *sim = &sims[sid];
	double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
	i = sim->ilist[ii];
	itype = map[sim->type[i]];
	for (int jj=0;jj<jnum;jj++){
		Sik[jj]=1;
		Bij[jj]=true;
		dSikx[jj]=0;
		dSiky[jj]=0;
		dSikz[jj]=0;
	}
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkx[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijky[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkz[jj*jnum+kk]=0;
	for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
		//if (Bij[kk]==false){continue;}
		ktype = tn[kk];
		delx2 = xn[kk];
		dely2 = yn[kk];
		delz2 = zn[kk];
		rik = delx2*delx2+dely2*dely2+delz2*delz2;
		if (rik>cutmax*cutmax){
			//Bij[kk]= false;
			continue;
		}
		for (jj=0;jj<jnum;jj++){
			if (jj==kk){continue;}
			//if (Bij[jj]==false){continue;}
			jtype = tn[jj];
			delx = xn[jj];
			dely = yn[jj];
			delz = zn[jj];
			rij = delx*delx+dely*dely+delz*delz;
			if (rij>cutmax*cutmax){
				//Bij[jj] = false;
				continue;
			}
			delx3 = delx2-delx;
			dely3 = dely2-dely;
			delz3 = delz2-delz;
			rjk = delx3*delx3+dely3*dely3+delz3*delz3;
			if (rik+rjk-rij<1e-13){continue;}//bond angle > 90 degrees
			if (rik+rij-rjk<1e-13){continue;}//bond angle > 90 degrees
			double Cmax = screening_max[itype*nelements*nelements+jtype*nelements+ktype];
			double Cmin = screening_min[itype*nelements*nelements+jtype*nelements+ktype];
			double temp1 = rij-rik+rjk;
			Cn = temp1*temp1-4*rij*rjk;
			temp1 = rij-rjk;
			Cd = temp1*temp1-rik*rik;
			Cijk = Cn/Cd;
			C = (Cijk-Cmin)/(Cmax-Cmin);
			if (C>=1){continue;}
			else if (C<=0){
				//Bij[kk]=false;
				Sik[kk]=0.0;
				dSikx[kk]=0.0;
				dSiky[kk]=0.0;
				dSikz[kk]=0.0;
				break;
			}
			dC = Cmax-Cmin;
			dC *= dC;
			dC *= dC;
			temp1 = 1-C;
			temp1 *= temp1;
			temp1 *= temp1;
			Sijk = 1-temp1;
			Sijk *= Sijk;
			Dij = 4*rik*(Cn+4*rjk*(rij+rik-rjk))/Cd/Cd;
			Dik = -4*(rij*Cn+rjk*Cn+8*rij*rik*rjk)/Cd/Cd;
			Djk = 4*rik*(Cn+4*rij*(rik-rij+rjk))/Cd/Cd;
			temp1 = Cijk-Cmax;
			double temp2 = temp1*temp1;
			dfc = 8*temp1*temp2/(temp2*temp2-dC);
			Sik[kk] *= Sijk;
			dSijkx[kk*jnum+jj] = dfc*(delx*Dij-delx3*Djk);
			dSikx[kk] += dfc*(delx2*Dik+delx3*Djk);
			dSijky[kk*jnum+jj] = dfc*(dely*Dij-dely3*Djk);
			dSiky[kk] += dfc*(dely2*Dik+dely3*Djk);
			dSijkz[kk*jnum+jj] = dfc*(delz*Dij-delz3*Djk);
			dSikz[kk] += dfc*(delz2*Dik+delz3*Djk);
		}
	}
	}
}

//treats # as starting a comment to be ignored.
int PairRANN::count_words(char *line){
	return count_words(line,": ,\t_\n");
}

int PairRANN::count_words(char *line,char *delimiter){
	int n = strlen(line) + 1;
	char copy[n];
	strncpy(copy,line,n);
	char *ptr;
	if ((ptr = strchr(copy,'#'))) *ptr = '\0';
	if (strtok(copy,delimiter) == NULL) {
		return 0;
	}
	n=1;
	while ((strtok(NULL,delimiter))) n++;
	return n;
}

void PairRANN::errorf(const std::string &file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file.c_str(),line,message);
	exit(1);
}

void PairRANN::errorf(char *file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file,line,message);
	exit(1);
}

void PairRANN::errorf(const char *message){
	//see about adding message to log file
	std::cout<<message;
	std::cout<<"\n";
	exit(1);
}


int PairRANN::factorial(int n) {
   if ((n==0)||(n==1))
      return 1;
   else
      return n*factorial(n-1);
}

std::vector<std::string> PairRANN::tokenmaker(std::string line,std::string delimiter){
	int nwords = count_words(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	char **words=new char *[nwords+1];
	nwords = 0;
	words[nwords++]=strtok(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	while ((words[nwords++] = strtok(NULL,const_cast<char *>(delimiter.c_str())))) continue;
	nwords--;
	std::vector<std::string> linev;
	for (int i=0;i<nwords;i++){
		linev.emplace_back(words[i]);
	}
	delete [] words;
	return linev;
}
