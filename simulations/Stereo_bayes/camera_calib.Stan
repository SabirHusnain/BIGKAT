functions{

	vector cross(vector a, vector b){
		/*Return the cross product of vectors a and b*/

		vector[3] output;

		output[1] = a[2] * b[3] - a[3] * b[2];
		output[2] = a[3] * b[1] - a[1] * b[3];
		output[3] = a[1] * b[2] - a[2] * b[1];
	    
	    return output;

	}


	vector rotate_rodrig(vector v, vector k){
		/*Rotate vector v around vector k.
		  The length of k encodes the magnitude of the rotation*/

		real theta;

		theta = sqrt(dot_product(k,k));
		return v * cos(theta) + cross(k,v) * sin(theta) + k * dot_product(k, v) * (1-cos(theta));

	}


	matrix position_object(matrix board, vector rod, vector tran){
		/*position a chessboard in the world by rotatiting it with rod and translating with tran */

		matrix[rows(board), cols(board)] positioned_board;

		for (point in 1:rows(board)){

			positioned_board[point] = (rotate_rodrig(board[point]', rod) + tran)';

		}

		return positioned_board;


	}

	matrix project_object(matrix board, matrix M){
		/*Project an object onto the camera using matrix M*/

		matrix[rows(board), cols(board)] projected_image_homog;
		matrix[rows(board), cols(board)] projected_image;

		for (point in 1:rows(board)){

			projected_image_homog[point] = (M * board[point]')';

			projected_image[point] = projected_image_homog[point] / projected_image_homog[point, 3];

		}

		return projected_image;

	}

}

data {
	
	int J; //Number of points in chessboard

	matrix[J, 3] chessboard;


	int P; //Number of images 
	vector[3] rod[P];
	vector[3] trans[P];
	matrix[J, 3] images[P]; 
	matrix[J, 3] images2[P]; 

}



parameters{
	
	
	real<lower = 0> sigma;
	real<lower = 0> sigma2;

	real<lower = 0> fx; 
	real<lower = 0> fy;
	real <lower = 0, upper =1280> cx; 
	real <lower = 0, upper = 720> cy; 

	real<lower = 0> fx2; 
	real<lower = 0> fy2;
	real <lower = 0, upper =1280> cx2; 
	real <lower = 0, upper = 720> cy2; 

	vector[3] R; //Camera2 rotation
	vector[3] T; //Camera2 translation


	//Rotation parameters (these should be less than a whole rotation)
	//real<lower = -pi(), upper = pi()> rod_x[P];
	//real<lower = -pi(), upper = pi()> rod_y[P];
	//real<lower = -pi(), upper = pi()> rod_z[P];

	//real trans_x[P];
	//real trans_y[P];
	//real<lower = 0> trans_z[P];

	
}

transformed parameters{
	
	//vector[3] rod[P]; //Rotation parameters
	//vector[3] trans[P]; //Translation parameters

	matrix[3,3] M1;
	matrix[3,3] M2;

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

	//Camera matrix 2
	M2[1,1] = fx2;
	M2[1,2] = 0;
	M2[1,3] = cx2;

	M2[2,1] = 0;
	M2[2,2] = fy2;
	M2[2,3] = cy2;

	M2[3,1] = 0;
	M2[3,2] = 0;
	M2[3,3] = 1;

	//Rotation and translation vectors

	//for (img in 1:P){

		//rod[img, 1] = rod_x[img];
		//rod[img, 2] = rod_y[img];
		//rod[img, 3] = rod_z[img];

		//trans[img, 1] = trans_x[img];
		//trans[img, 2] = trans_y[img];
		//trans[img, 3] = trans_z[img];

	//}

}


model{

	//trans_x ~ normal(500, 2000);
	//trans_y ~ normal(500, 2000);
	//trans_z ~ normal(500, 2000);



	for (img in 1:P){

		matrix[J, 3] new_object;
		matrix[J, 3] projection1;

		matrix[J, 3] new_object2;
		matrix[J, 3] projection2;

		new_object = position_object(chessboard, rod[img], trans[img]); //Position object in space
		projection1 = project_object(new_object, M1); //Project image to screen 1


		new_object2 = position_object(new_object, R, T);
		projection2 = project_object(new_object2, M2); //Project image to screen 1



		to_vector(images[img, 1:J, 1:2]) ~ normal(to_vector(projection1[1:J, 1:2]), sigma);
		to_vector(images2[img, 1:J, 1:2]) ~ normal(to_vector(projection2[1:J, 1:2]), sigma2);

	}
	

}

