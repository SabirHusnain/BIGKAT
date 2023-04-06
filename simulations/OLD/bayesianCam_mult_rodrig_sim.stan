functions{
	
	vector cross(vector a, vector b){
		/*Return the cross product of vectors a and b*/

		vector[3] output;

		output[1] <- a[2] * b[3] - a[3] * b[2];
		output[2] <- a[3] * b[1] - a[1] * b[3];
		output[3] <- a[1] * b[2] - a[2] * b[1];
	    
	    return output;

	}

	
	vector rotate_rodrig(vector v, vector k){
		/*Rotate vector v around vector k.
		  The length of k encodes the magnitude of the rotation*/

		real theta;

		theta <- sqrt(dot_product(k,k));
		return v * cos(theta) + cross(k,v) * sin(theta) + k * dot_product(k, v) * (1-cos(theta));

	}

	vector mm2pixel(vector p, vector size, vector resolution){
		/*Convert from mm to screen coordinates*/
		vector[3] output;

		output[1] <- p[1] * resolution[1] / size[1] + resolution[1] / 2;
		output[2] <- p[2] * resolution[2] / size[2] + resolution[2] / 2;
		output[3] <- 1;

		return output;
	}	

}



data{

	int P; //Number of images
	int K; //Number of corners per image

	real images[P*K*2]; //Image data in homogenous coordiantes
	vector[3] chessboard[K]; //Kx3 chessboard points homogenous
	vector[2] size_;
	vector[2] resolution;
}



parameters{		

}

transformed parameters{

	
}


model{
	
	
}

generated quantities{
	

	vector[3] Q; 
	vector[P*K*2] Q_hat;
	int i;

	vector[3] rotations[P]; //Rotations vector (rodriguez for each image)
	vector[3] translations[P]; //Translation vector for each image
	
	matrix[3,3] M;	

	real<lower = 0> fx;
	real<lower = 0> fy;
	real cx;
	real cy;

	fx <- 4;
	fy <- 3.5;
	cx <- 0;
	cy <- 0;


	rotations[1][1] <- -0.6;
	rotations[1][2] <- 0.24;
	rotations[1][3] <- .3;

	rotations[2][1] <- -0.8;
	rotations[2][2] <- -0.5;
	rotations[2][3] <- 0.5;

	translations[1][1] <- -24.5*8/2;
	translations[1][2] <- -24.5*5/2;
	translations[1][3] <- 250;

	translations[2][1] <- 20;
	translations[2][2] <- -50;
	translations[2][3] <- 300;

	M[1,1] <- fx;
	M[1,2] <- 0;
	M[1,3] <- cx;
	M[2,1] <- 0;
	M[2,2] <- fy;
	M[2,3] <- cy;
	M[3,1] <- 0.0;
	M[3,2] <- 0.0;
	M[3,3] <- 1.0;	

	
	i <- 1;

	for (img in 1:P){
		
		for (pt in 1:K){

			Q <- M * (rotate_rodrig(chessboard[pt], rotations[img]) + translations[img]);				
			Q <- Q / Q[3]; //Divide through by w
			
			Q_hat[i:i+1] <- mm2pixel(Q, size_, resolution)[1:2]; //Convert to pixel coordinates
			
			i <- i + 2;		
			print(pt);
		}	

	}
	
}
