//std
#include <iostream>
#include <sstream>
#include <fstream>

//CUDA
#ifdef ENABLE_CUDA
	#include <cuda_runtime.h>
	#include <cusparse_v2.h>
	#include <cublas_v2.h>
	#include <helper_functions.h>
	#include <helper_cuda.h>
#endif

template <typename TYPE> class VAR{
private:
public:
	TYPE *h;
	#ifdef ENABLE_CUDA
	cublasHandle_t cublasHandle;
	cublasStatus_t cublasStatus;
	#endif
	int M;
	int N;
	int NumOfEle;
	bool TypeOfIndex;

	//Convert Index
	int ConvertIndex(int _row,int _col){
		int index;
		if(this->TypeOfIndex)
			index = _col + _row * this->N;
		else
			index = _row + _col * this->M;
		if(index<0 || index>(this->NumOfEle)){
			exit(EXIT_FAILURE);
		}
		return index;
	}

	//Product
	double operator % (VAR obj) {
		double ResDot=0.0;
		//for(int i=0;i<(this->NumOfEle);i++) res.h[i] -= obj.h[i];
		//Vector Only
		if(this->M==obj.M && this->N==obj.N){
			int NNN = this->M * this->N;
			#ifdef ENABLE_CUDA
				/* Calc on GPU */
				double *dA,*dB;
				checkCudaErrors(cudaMalloc((void **)&dA, sizeof(double)*NNN));
				checkCudaErrors(cudaMalloc((void **)&dB, sizeof(double)*NNN));
				cudaMemcpy(dA, this->h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				cudaMemcpy(dB, obj.h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				cublasDdot(cublasHandle, NNN, dA, 1, dB, 1, &ResDot);
				cudaError_t err;
				err=cudaFree(dA);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
				err=cudaFree(dB);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
			#else
				/* Calc on CPU */
				for(int i=0;i<NNN;i++)
					ResDot += this->h[i] * obj.h[i];
			#endif

		}else{
			std::cout << "Dimension Error!" << std::endl;
		}
		return ResDot;
	}
	//Minus
	VAR operator - (VAR obj) {
		VAR<TYPE> res(this->M,this->N);
		//for(int i=0;i<(this->NumOfEle);i++) res.h[i] -= obj.h[i];
		if(this->M==obj.M && this->N==obj.N){
			int NNN = this->M * this->N;
			#ifdef ENABLE_CUDA
				double alpha = -1.0;
				double *dA,*dB;
				checkCudaErrors(cudaMalloc((void **)&dA, sizeof(double)*NNN));
				checkCudaErrors(cudaMalloc((void **)&dB, sizeof(double)*NNN));
				cudaMemcpy(dA, this->h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				cudaMemcpy(dB, obj.h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				// dA = dA + dB
				cublasDaxpy(cublasHandle, NNN, &alpha, dB, 1, dA, 1);
				cudaMemcpy(res.h, dA , sizeof(double)*NNN, cudaMemcpyDeviceToHost);
				cudaError_t err;
				err=cudaFree(dA);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
				err=cudaFree(dB);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
			#else
				for(int i=0;i<NNN;i++)
					res.h[i] = this->h[i] - obj.h[i];
			#endif
		}else{
			std::cout << "Dimension Error!" << std::endl;
		}
		return res;
	}
	//Add
	VAR operator + (VAR obj) {
		VAR<TYPE> res(this->M,this->N);
		//for(int i=0;i<(this->NumOfEle);i++) res.h[i] -= obj.h[i];
		if(this->M==obj.M && this->N==obj.N){
			int NNN = this->M * this->N;
			#ifdef ENABLE_CUDA
				double alpha = +1.0;
				double *dA,*dB;
				checkCudaErrors(cudaMalloc((void **)&dA, sizeof(double)*NNN));
				checkCudaErrors(cudaMalloc((void **)&dB, sizeof(double)*NNN));
				cudaMemcpy(dA, this->h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				cudaMemcpy(dB, obj.h , sizeof(double)*NNN, cudaMemcpyHostToDevice);
				// dA = dA + dB
				cublasDaxpy(cublasHandle, NNN, &alpha, dB, 1, dA, 1);
				cudaMemcpy(res.h, dA , sizeof(double)*NNN, cudaMemcpyDeviceToHost);
				cudaError_t err;
				err=cudaFree(dA);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
				err=cudaFree(dB);
				if (err != cudaSuccess) std::cout << "error:" << cudaGetErrorString(err) << std::endl;
			#else
				for(int i=0;i<NNN;i++)
					res.h[i] = this->h[i] + obj.h[i];
			#endif
		}else{
			std::cout << "Dimension Error!" << std::endl;
		}
		return res;
	}
	//Mul
	VAR operator * (VAR obj) {
		/*
		VAR<TYPE> res = *this;
		for(int i=0;i<(this->NumOfEle);i++)
			res.h[i] *= obj.h[i];
		*/
		VAR<TYPE> res(this->M,obj.N);
		res.Zero();
		if( this->N == obj.M ){
			int M = this->M;
			int N = obj.N;
			int K = obj.M;
			#ifdef ENABLE_CUDA
				double *hA,*hB,*hC;
				double *dA,*dB,*dC;
				hA = this->h;
				hB = obj.h;
				hC = res.h;
				checkCudaErrors(cudaMalloc((void **)&dA, sizeof(double)*M*K));
				checkCudaErrors(cudaMalloc((void **)&dB, sizeof(double)*K*N));
				checkCudaErrors(cudaMalloc((void **)&dC, sizeof(double)*M*N));
				cudaMemcpy(dA, hA , sizeof(double)*M*K, cudaMemcpyHostToDevice);
				cudaMemcpy(dB, hB , sizeof(double)*K*N, cudaMemcpyHostToDevice);
				cudaMemcpy(dC, hC , sizeof(double)*M*N, cudaMemcpyHostToDevice);
				double alpha = 1.0 , beta = 0.0;
				cublasDgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,&alpha,dA,M,dB,K,&beta,dC,M);
				cudaMemcpy(hC, dC , sizeof(double)*M*N, cudaMemcpyDeviceToHost);
			#else
				for(int ii=0;ii<M;ii++){
					for(int jj=0;jj<N;jj++){
						double temp = 0.0;
						for(int kk=0;kk<K;kk++)
							temp+= this->h[ this->ConvertIndex(ii,kk) ] * obj.h[ obj.ConvertIndex(kk,jj) ];
						res.h[ res.ConvertIndex(ii,jj) ] = temp;
					}
				}
			#endif
		}else{
			std::cout << "Dimension Error!" << std::endl;
		}
		return res;
	}

	TYPE& operator () (int _row) {
		int index = ConvertIndex(_row,0);
		return this->h[index];
	}
	TYPE& operator () (int _row,int _col) {
		int index = ConvertIndex(_row,_col);
		return this->h[index];
	}
	VAR(){ }
	VAR(int _M,int _N){
		this->M = _M;
		this->N = _N;
		this->NumOfEle = _M * _N;
		this->TypeOfIndex = false;
		#ifdef ENABLE_CUDA
			this->cublasHandle = 0;
			this->cublasStatus = cublasCreate(&this->cublasHandle);
			if (checkCudaErrors(this->cublasStatus)) exit(EXIT_FAILURE);
		#endif
		this->h = new TYPE[ this->NumOfEle ];
		return;
	}
	VAR(const VAR& obj){
		this->M = obj.M;
		this->N = obj.N;
		this->NumOfEle = obj.NumOfEle;
		this->TypeOfIndex = obj.TypeOfIndex;
		#ifdef ENABLE_CUDA
			this->cublasHandle = obj.cublasHandle;
			this->cublasStatus = obj.cublasStatus;
		#endif
		this->h = new TYPE[ this->NumOfEle ];
		for(int i=0;i<(this->NumOfEle);i++) this->h[i] = obj.h[i];
		return;
	}
	~VAR(){
		//delete[] this->h;
		//cublasDestroy(this->cublasHandle);
		return;
	}

	void Ones(){
		for(int i=0;i<(this->NumOfEle);i++)
			this->h[i] = (TYPE)1;
		return;
	}
	void Zero(){
		for(int i=0;i<(this->NumOfEle);i++)
			this->h[i] = (TYPE)0;
		return;
	}

	void SaveCsv(const char *fn){
		std::ofstream ofs(fn);
		for(int i=0;i<(this->M);i++){
		for(int j=0;j<(this->N);j++){
			int index = this->ConvertIndex(i,j);
			ofs << this->h[ index ];
			if(j==(this->N)-1) ofs << std::endl; else ofs << ",";
		}		
		}
		ofs.close();
		return;
	}	

	/*
	void Test(){
		int M = 1000;
		int N = 1;
		int K = 1000;
		double *hA,*hB,*hC;
		double *dA,*dB,*dC;
		hA = new double [ M*K ];
		hB = new double [ K*N ];
		hC = new double [ M*N ];
		checkCudaErrors(cudaMalloc((void **)&dA, sizeof(double)*M*K));
		checkCudaErrors(cudaMalloc((void **)&dB, sizeof(double)*K*N));
		checkCudaErrors(cudaMalloc((void **)&dC, sizeof(double)*M*N));
		for(int i=0;i<M*K;i++) hA[i] = 1.0;
		for(int i=0;i<K*N;i++) hB[i] = 1.0;
		for(int i=0;i<M*N;i++) hC[i] = 0.0;
		cudaMemcpy(dA, hA , sizeof(double)*M*K, cudaMemcpyHostToDevice);
		cudaMemcpy(dB, hB , sizeof(double)*K*N, cudaMemcpyHostToDevice);
		cudaMemcpy(dC, hC , sizeof(double)*M*N, cudaMemcpyHostToDevice);
		double alpha = 1.0 , beta = 0.0;
		cublasDgemm(cublasHandle,CUBLAS_OP_N,CUBLAS_OP_N,M,N,K,&alpha,dA,M,dB,K,&beta,dC,M);
		cudaMemcpy(hC, dC , sizeof(double)*M*N, cudaMemcpyDeviceToHost);
		for(int i=0;i<10;i++) std::cout << hC[i] << std::endl;
		delete[] hA;
		delete[] hB;
		delete[] hC;
		return;
	}
	*/

};

int main() {

	VAR<double> A(2,2);
	VAR<double> B(2,2);
	VAR<double> C;

	A(0,0) = +1.0;
	A(0,1) = -1.0;
	A(1,0) = -2.0;
	A(1,1) = +3.0;

	B(0,0) = +1.0;
	B(0,1) = +2.0;
	B(1,0) = +3.0;
	B(1,1) = +4.0;

	C = A + B;
	C.SaveCsv("a.csv");

	return 0;
}
