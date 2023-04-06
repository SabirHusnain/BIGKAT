functions{

	vector cross(vector a, vector b){
		/*Return the cross product of vectors a and b*/

		vector[3] output;

		output[1] = a[2] * b[3] - a[3] * b[2];
		output[2] = a[3] * b[1] - a[1] * b[3];
		output[3] = a[1] * b[2] - a[2] * b[1];
	    
	    return output;

	}

	vector rotate_rodrig(vector v, vector k, real theta){
		/*Rotate vector v around vector k.
		  The length of k encodes the magnitude of the rotation*/

		
		return v * cos(theta) + cross(k,v) * sin(theta) + k * dot_product(k, v) * (1-cos(theta));

	}




}

data {
	
	int J; //Number of points in chessboard

	matrix[J, 3] chessboard;
	int P; //Number of images 
	matrix[J, 3] images[P]; 

	real resolution[2];

	//real cx;
	//real cy;
	//real fx;
	//real fy;

	//vector[3] rod[P];
	vector[3] trans[P];
	real sigma;
}

transformed data{
	
	real cx;
	real cy;

	cx = resolution[1] / 2.0;
	cy = resolution[2] / 2.0;
}

parameters{
	
	
	//real<lower = 0> sigma;	
	real<lower = 0> fx; 
	real<lower = 0> fy;
	//real <lower = 0, upper = 1280> cx; 
	//real <lower = 0, upper = 720> cy; 



	//real trans_x[P];
	//real trans_y[P];
	//real<lower = 0> trans_z[P];

	simplex[3] rod[P];
	real<lower = -pi()/2, upper= pi()/2> mag[P];	
	
}


transformed parameters{
	
	//vector[3] rod[P]; //Rotation parameters
	//vector[3] trans[P]; //Translation parameters

	matrix[3,3] M1;
	//matrix[3,3] RT[P];

	//Camera matrix
	M1[1,1] = fx;
	M1[1,2] = 0;
	M1[1,3] = cx;

	M1[2,1] = 0;
	M1[2,2] = fy;
	M1[2,3] = cy;

	M1[3,1] = 0;
	M1[3,2] = 0;
	M1[3,3] = 1;

	//Rotation and translation vectors

	for (img in 1:P){

		//trans[img][1] = trans_x[img];
		//trans[img][2] = trans_y[img];
		//trans[img][3] = trans_z[img];

		
	}

	
}


model{

	//sigma ~ cauchy(0, 5);

	fx ~ normal(896, 200);
	fy ~ normal(504, 200);


	for (img in 1:P){
		matrix[J, 3] all_projection;

		for (point in 1:rows(chessboard)){

			vector[3] projection;

			projection = M1 * (rotate_rodrig(chessboard[point]', rod[img], mag[img]) + trans[img]);
			projection = projection / projection[3];

			all_projection[point] = projection';

			
		}

	to_vector(images[img, 1:J,1:2]) ~ normal(to_vector(all_projection[1:J, 1:2]), sigma);

	}
}

