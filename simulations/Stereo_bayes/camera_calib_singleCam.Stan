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
	matrix[J, 3] images[P]; 

	real resolution[2];

	//real cx;
	//real cy;
	//real fx;
	//real fy;

	//vector[3] rod[P];
	matrix[3,3] RT_dat[P];
}

transformed data{
	
	real cx;
	real cy;

	cx = resolution[1] / 2.0;
	cy = resolution[2] / 2.0;
}

parameters{
	
	
	real<lower = 0> sigma;	
	real<lower = 0> fx; 
	real<lower = 0> fy;
	//real <lower = 0, upper = 1280> cx; 
	//real <lower = 0, upper = 720> cy; 


	//unit_vector[3] T[P];

	real trans_x[P];
	real trans_y[P];
	//real<lower = 0> trans_z[P];


	vector<lower= -pi(), upper = pi()>[6] R[P]; //Rotation paramers

	
	
}

transformed parameters{
	
	//vector[3] rod[P]; //Rotation parameters
	//vector[3] trans[P]; //Translation parameters

	matrix[3,3] M1;
	matrix[3,3] RT[P];

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
		
		RT[img][1,1] = R[img][1];
		RT[img][1,2] = R[img][2];
		RT[img][1,3] = trans_x[img];

		RT[img][2,1] = R[img][3];
		RT[img][2,2] = R[img][4];
		RT[img][2,3] = trans_y[img];

		RT[img][3,1] = R[img][5];
		RT[img][3,2] = R[img][6];
		RT[img][3,3] = RT_dat[img][3,3];

	}
}


model{

	sigma ~ cauchy(0, 5);
	//cx ~ normal(resolution[1] /2.0, 20); 
	//cy ~ normal(resolution[2] /2.0, 20);

	fx ~ normal(896, 200);
	fy ~ normal(504, 200);

	//trans_z ~ normal(500, 800);

	for (img in 1:P){

		R[img] ~ von_mises(0.0, 2.0);
	}
	




	for (img in 1:P){
		matrix[J, 3] all_projection;

		for (point in 1:rows(chessboard)){

			vector[3] projection;

			projection = M1 * RT[img] *    chessboard[point]';
			projection = projection / projection[3];

			all_projection[point] = projection';

			


		}

	to_vector(images[img, 1:J,1:2]) ~ normal(to_vector(all_projection[1:J, 1:2]), sigma);

	}
}

