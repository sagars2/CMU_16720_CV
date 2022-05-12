import numpy as np
import cv2


def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    
    A = []
    for i in range(len(x1)):
        A.append([-x2[i,0], -x2[i,1],-1, 0, 0, 0, x2[i,0]*x1[i,0], x1[i,0]*x2[i,1], x1[i,0]])
        A.append([0, 0, 0, -x2[i,0], -x2[i,1], -1, x1[i,1]*x2[i,0], x1[i,1]*x2[i,1],x1[i,1]])
        
    A = np.array(A)
    
    U,Sigma,V_transpose = np.linalg.svd(A.T @ A)

    H2to1 = V_transpose[-1,:]
    H2to1 = np.reshape((H2to1),(3,3))

    return H2to1


def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    x1_x = np.mean(x1[:,0])
    x1_y = np.mean(x1[:,1])
    x2_x = np.mean(x2[:,0])
    x2_y = np.mean(x2[:,1])

    x1_centroid = [x1_x,x1_y]
    x2_centroid = [x2_x,x2_y]
    
    #Shift the origin of the points to the centroid
    x1_shifted = x1-x1_centroid
    x2_shifted = x2-x2_centroid

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    
    # x1_norm = np.linalg.norm(x1_centroid)
    # x2_norm = np.linalg.norm(x2_centroid)
    
    
    # # x1_scaled = x1_norm*sqrt(2)/
    # if x1_norm > x2_norm:
    #     x1_scaled = np.testing.assert_equal(x1_norm,np.sqrt(2))
    #     x2_scaled = (x2_norm * np.sqrt(2))/x1_norm
    # else:
    #     x2_scaled = np.testing.assert_equal(x2_norm,np.sqrt(2))
    #     x1_scaled = (x1_norm*np.sqrt(2))/x2_norm
    
    # print(x1_scaled)
    # print(x2_scaled)
    # print(x1_norm)
    #Similarity transform 1
    
    x1_norm = np.sqrt((x1_shifted[:,0]**2)+(x1_shifted[:,1]**2))
    x2_norm = np.sqrt((x2_shifted[:,0]**2)+(x2_shifted[:,1]**2))
    
    x1_scaled = np.sqrt(2)/np.max(x1_norm)
    x2_scaled = np.sqrt(2)/np.max(x2_norm)
    
    
    A1 = np.eye(3)*x1_scaled
    A2 = np.array([[1,0,-x1_x],[0,1,-x1_y],[0,0,1]])
    
    T1 = np.matmul(A1,A2)

    #Similarity transform 2
    A3 = np.eye(3)*x2_scaled
    A4 = np.array([[1,0,-x2_x],[0,1,-x2_y],[0,0,1]])

    T2 = np.matmul(A3,A4)
    #Compute homography
    H = computeH(x1_shifted,x2_shifted)

    #Denormalization
    
    H2to1 = np.matmul(np.linalg.inv(T1),H,T2)
    
    return H2to1


# def computeH_ransac(locs1, locs2, opts):
    # #Q2.2.3
    # #Compute the best fitting homography given a list of matching points
    # max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    
    # #converting locs from (y,x) to (x,y)
    # locs1 = locs1[:,[1,0]]
    # locs2 = locs2[:,[1,0]]    
    
    # # print('Locs1',locs1.shape)
    # bestH2to1 = np.zeros((3,3))
    # inliers = np.zeros(len(locs1))
    # # inliers = []
    # inlier_tol = 2
    # for i in range(max_iters):
    #     #generating 4 point pairs
    #     point_pairs = np.random.choice(locs1.shape[0],4,False)         
    #     H = computeH_norm(locs1[point_pairs,:], locs2[point_pairs,:]) #outputs a 3 by 3 array
        
        
    #     inliers = np.zeros(locs1.shape[0])
    #     c_inlier = [0]*len(inliers)
    #     for s in range(len(inliers)):
    #         locs1_pred = np.hstack((locs1, np.ones((locs1.shape[0], 1))))
    #         locs2_pred = np.hstack((locs2, np.ones((locs2.shape[0], 1))))
            
    #         l1 = np.hstack((locs1,np.ones((locs1.shape[0], 1))))
    #         l2 = np.hstack((locs2,np.ones((locs2.shape[0],1))))
            
    #         locs1_pred = np.matmul(H,locs2_pred[s])
    #         locs1_pred = locs1_pred/locs1_pred[2] 
            
    #         locs2_pred = np.matmul(H,locs2_pred[s])
    #         locs2_pred = locs2_pred/locs2_pred[2]
            
    #         err_l1 = np.linalg.norm(locs1_pred - l1)
    #         err_l2 = np.linalg.norm(locs2_pred - l2)
    #         #computes the l2 norm
    #         if err_l1 < inlier_tol:
    #             c_inlier[s] = 1
    #             print("condition satisfied")
    #         else:
    #             print("condition not satisfied")
    #         s+=1
        
    #     # if np.sum(c_inlier)>=np.sum(inliers):
            
    #     bestH2to1 = np.where(np.sum(c_inlier)>=np.sum(inliers),H)
    #     inliers = np.where(np.sum(c_inlier)>=np.sum(inliers),c_inlier)

    
    # return bestH2to1, inliers

def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier
    #converting locs from (y,x) to (x,y)
    locs1 = locs1[:,[1,0]]
    locs2 = locs2[:,[1,0]]    
    
    l1 = np.hstack((locs1,np.ones((locs1.shape[0], 1))))
    l2 = np.hstack((locs2,np.ones((locs2.shape[0], 1))))
    
    inlier_no = 0
    
    for i in range(max_iters):
        np.random.seed(seed=None)
        point_pairs = np.random.choice(locs1.shape[0],4,False)
        locs1_pred = locs1[point_pairs,:]
        locs2_pred = locs2[point_pairs,:]
        
        H = computeH(locs1_pred,locs2_pred)        
        l_1 = (np.matmul(H,l2.T))
        last_col = l_1[2,:].T
        l_1 = l_1.T/last_col[:,None]
        
        # l_1 = np.transpose(l_1.T/(l_1[2,:])[:,None])


        inliers = np.zeros(locs1.shape[0])
        # c_inlier = np.zeros(len(inliers))
        
        err_l1 = np.linalg.norm(l1 - l_1,axis=1)
        inliers = np.sum(err_l1<inlier_tol)
        
       
        # for j in range(len(inliers)):
        # #     err_l1 = np.linalg.norm(l1 - l_1)
        #     if err_l1 < inlier_tol:
        #         c_inlier[i] = 1
        #         i+=1

        if inliers>=inlier_no:
            bestH2to1 = H
            # inliers = c_inlier
            inlier_no = inliers

        
        # # err_l2 = np.linalg.norm(l_2 - l2)
    
    return bestH2to1, inliers
    
    

def compositeH(H2to1, template, img):

    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.
    #Create mask of same size as template
    
    h = img.shape[0]
    w = img.shape[1]

    mask = np.ones((template.shape))
    #Warp mask by appropriate homography
    mask = cv2.warpPerspective(mask, np.linalg.inv(H2to1), (w,h))
    
    
   
    # idx = np.nonzero(mask.T)
    #Warp template by appropriate homography
    warp_homography = cv2.warpPerspective(template, np.linalg.inv(H2to1), (w,h))
    idx = mask == 0
    idx = idx.astype(int)
    #Use mask to combine the warped template and the image
    
    composite_img = idx*img+warp_homography
    
    # warp_homography = warp_homography.T


    # img[idx[0],0] = warp_homography[idx[0],0]
    # img[idx[1],1] = warp_homography[idx[1],1]
    # img[idx[2],2] = warp_homography[idx[2],2]
    
    # composite_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return composite_img


