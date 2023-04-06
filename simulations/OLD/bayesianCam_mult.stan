functions{
	
	matrix get_RT_matrix(real r1,real r2,real r3, real t1,real t2,real t3){

		matrix[3,3] Rx;
		matrix[3,3] Ry;
		matrix[3,3] Rz;
		matrix[3,3] RT;

		vector[3] T;

		Rx[1,1] <- 1;
		Rx[1,2] <- 0;
		Rx[1,3] <- 0;
		Rx[2,1] <- 0;
		Rx[2,2] <- cos(r1);
		Rx[2,3] <- -sin(r1);
		Rx[3,1] <- 0;
		Rx[3,2] <- sin(r1);
		Rx[3,3] <- cos(r1);

		Ry[1,1] <- cos(r2);
		Ry[1,2] <- 0;
		Ry[1,3] <- sin(r2);
		Ry[2,1] <- 0;
		Ry[2,2] <- 1;
		Ry[2,3] <- 0;
		Ry[3,1] <- -sin(r2);
		Ry[3,2] <- 0;
		Ry[3,3] <- cos(r2);

		Rz[1,1] <- cos(r3);
		Rz[1,2] <- -sin(r3);
		Rz[1,3] <- 0;
		Rz[2,1] <- sin(r3);
		Rz[2,2] <- cos(r3);
		Rz[2,3] <- 0;
		Rz[3,1] <- 0;
		Rz[3,2] <- 0;
		Rz[3,3] <- 1;

		T[1] <- t1;
		T[2] <- t2;
		T[3] <- t3;

		RT <- Rx * Ry * Rz;
		RT[1,3] <- T[1];
		RT[2,3] <- T[2];
		RT[3,3] <- T[3];


		return RT;

	}


}

data{

	int P; //Number of images
	int K; //Number of corners per image

	real images[P*K*2]; //Image data in homogenous coordiantes
	vector[3] chessboardH[K]; //Kx3 chessboard points homogenous
	vector[3] k_vec;
}


parameters{
		
	real<lower = 0> sigma;	
	real<lower = 0> fx;
	real<lower = 0> fy;
	real cx;
	real cy;

	real t1[P];
	real t2[P];
	//real<lower = 0> t3[P];	

	real r1[P];
	real r2[P];
	real r3[P];
	


}

transformed parameters{

	matrix[3,3] RT[P];
	matrix[3,3] M;	

	//real<lower = 0> fx;
	//real<lower = 0> fy;
	//real cx;
	//real cy;
	
	//real t1[P];
	//real t2[P];
	real<lower = 0> t3[P];	

	//t1[1] <- 200;
	//t2[1] <- 80;
	t3[1] <- 400;

	//t1[2] <- 50;
	//t2[2] <- 50;
	//t3[2] <- 20;

	//t1[3] <- 60;
	//t2[3] <- 10;
	//t3[3] <- 355;


	//r1[1] <- pi()/5.0;
	//r2[1] <- pi()/4.0;
	//r3[1] <- 0;
	


	M[1,1] <- fx;
	M[1,2] <- 0;
	M[1,3] <- cx;
	M[2,1] <- 0;
	M[2,2] <- fy;
	M[2,3] <- cy;
	M[3,1] <- 0.0;
	M[3,2] <- 0.0;
	M[3,3] <- 1.0;


	for (img in 1:P){
		RT[img] <- get_RT_matrix(r1[img],r2[img],r3[img],t1[img],t2[img],t3[img]);
	}

}

model{
	
	vector[3] Q; 
	vector[P*K*2] Q_hat;
	int i;

	sigma ~ cauchy(0, 5);;	
	fx ~ cauchy(0, 15);
	fy ~ cauchy(0, 15);
	cx ~ normal(0, 15);
	cy ~ normal(0, 15);

	r1 ~ normal(0, 1);
	r2 ~ normal(0, 1);
	r3 ~ normal(0, 1);

	t1 ~ normal(200, 500);
	t2 ~ normal(200, 500);
	//t3 ~ normal(300, 500);
	
	i <- 1;
	for (img in 1:P){
		
		for (pt in 1:K){

			Q <- M * RT[img] * chessboardH[pt];  //Transform
			
			Q <- Q / Q[3]; //Divide through by w
			
			Q_hat[i] <- (Q .* k_vec)[1];
			Q_hat[i+1] <- (Q .* k_vec)[2];
			i <- i + 2;		
			
		}	


	}
	images ~ normal(Q_hat, sigma);

}