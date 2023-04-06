data {
	
	int J; //Number of points in chessboard
	matrix[J, 3] chessboard;
	int P; //Number of images 
	matrix[J, 3] images[P]; 

}


parameters{

	matrix[3,3] H[P];
	real sigma;

}


transformed parameters{

}


model{

	for (img in 1:P){

		for (corner in 1:J){

			images[img, corner] ~ normal( (H[img] * chessboard[corner]')', sigma);

		}

	}
}

generated quantities{
	
	matrix[2*P, 6] V;
	vector[6] v[P, 3,3];
	int count;

	for (img in 1:P){

		for (i in 1:3){

			for (j in 1:3){

				v[img, i, j, 1] = H[img, i, 1] * H[img,j,1];
				v[img, i, j, 2] = H[img,i,1] * H[img,j, 2] + H[img,i,2] * H[img,j,1];
				v[img, i, j, 3] = H[img,i,2] * H[img,j,2];
				v[img, i, j, 4] = H[img,i,3] * H[img,j,1] + H[img,i,1] * H[img,j,3];
				v[img, i, j, 5] = H[img,i,3] * H[img,j,2] + H[img,i,2] * H[img,j,3];
				v[img, i, j, 6] = H[img,i,3] * H[img,j,3];
			}
		}

		
	}

	count = 1;
	for (img in 1:P){

		V[count] = v[img,1,2]'; 
		V[count+1] = (v[img,1,1] - v[img, 2,2])'; 
		count = count + 2;
	}


}
