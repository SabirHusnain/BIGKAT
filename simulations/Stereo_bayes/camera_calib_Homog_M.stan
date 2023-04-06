
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

	matrix project_object(vector s, matrix board, matrix M){
		/*Project an object onto the camera using matrix M*/

		matrix[rows(board), cols(board)] projected_image_homog;
		matrix[rows(board), cols(board)] projected_image;

		for (point in 1:rows(board)){

			projected_image_homog[point] = s[point] * (M * board[point]')';

			

		}

		return projected_image;

	}

}


data {
	
	int J; //Number of points in chessboard
	matrix[J, 3] chessboard;
	int P; //Number of images 
	matrix[J, 3] images[P]; 

}


parameters{

	matrix[3,3] H[P];
	real sigma;
	vector<lower = 0>[J] s[P];

	real<lower = 0> fx; 
	real<lower = 0> fy;
	real <lower = 0, upper =1280> cx; 
	real <lower = 0, upper = 720> cy; 

	matrix[3,3] RT[P];

	
	real<lower = -pi(), upper = pi()> rod_x[P];
	real<lower = -pi(), upper = pi()> rod_y[P];
	real<lower = -pi(), upper = pi()> rod_z[P];

	real trans_x[P];
	real trans_y[P];
	real<lower = 0> trans_z[P];


}


transformed parameters{

	matrix[3,3] M;
	vector[3] rod[P]; //Rotation parameters
	vector[3] trans[P]; //Translation parameters

	//Camera matrix
	M[1,1] = fx;
	M[1,2] = 0;
	M[1,3] = cx;

	M[2,1] = 0;
	M[2,2] = fy;
	M[2,3] = cy;

	M[3,1] = 0;
	M[3,2] = 0;
	M[3,3] = 1;

	for (img in 1:P){

		rod[img, 1] = rod_x[img];
		rod[img, 2] = rod_y[img];
		rod[img, 3] = rod_z[img];

		trans[img, 1] = trans_x[img];
		trans[img, 2] = trans_y[img];
		trans[img, 3] = trans_z[img];

	}

}


model{

	for (img in 1:P){

		matrix[J, 3] new_object;
		vector[3] projection;

		vector[ 3] new_chessboard;

		for (point in 1:J){

			new_chessboard = (rotate_rodrig(chessboard[point]', rod[img]) + trans[img]);

			projection = s[img, point] * M * new_chessboard;

			images[img, point] ~ normal(projection, sigma);


		}
		
	}
}

