import os
import sys
import cv2
import fnmatch
import numpy as np
import mahotas as mh 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

########################################################################################

def Check(Bin_Names,Bin_Start,Bin_Stop):
    if len(Bin_Names)!=len(Bin_Start)!=len(Bin_Stop):
        print('WARNING.  Bin list sizes are not of equal length')  
    
########################################################################################    
    
def Load_First(dpath,file):
    
    #Upoad file
    fpath = dpath + "/" + file
    print('file: '+ fpath)
    cap = cv2.VideoCapture(fpath)

    #Get maxiumum frame of file. Note that this is updated later if fewer frames detected
    cap_max = int(cap.get(7)) #7 is index of total frames
    print('total frames: ' + str(cap_max))

    #Set first frame to be displayed
    cap.set(1,0) #first index references frame property, second specifies next frame to grab

    #Initialize ycrop in the event that it is not subsequently set to something other than 0
    ycrop = 0

    #Load first frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cap.release() 
    
    return(fpath,gray)

########################################################################################

def Measure_Motion(fpath,ycrop,mt_cutoff,SIGMA):
    
    #Upoad file
    cap = cv2.VideoCapture(fpath)

    #Get maxiumum frame of file. Note that this is updated later if fewer frames detected
    cap_max = int(cap.get(7)) #7 is index of total frames

    #Set first frame to be grabbed
    cap.set(1,0) #first index references frame property, second specifies next frame to grab

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_new = frame_new[ycrop:,:]
    frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA)
    
    #Initialize vector to store motion values in
    Motion = np.zeros(cap_max)

    #Loop through frames to detect frame by frame differences
    for x in range (1,cap_max):

        #Reset old frame
        frame_old = frame_new

        #Attempt to load next frame
        ret, frame = cap.read()
        if ret == True:

            #Reset new frame and process
            frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_new = frame_new[ycrop:,:]
            frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA) # used to reduce influence of jitter from one frame to the next

            #Calculate difference between frames
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = frame_dif > mt_cutoff
            frame_cut = frame_cut.astype('uint8')

            #Assign difference to array
            Motion[x]=np.sum(frame_cut)

        else: 
            #if no frame is detected
            cap_max = (x-1) #Reset max frame to last frame detected
            Motion = Motion[:cap_max] #Amend length of motion vector
            break
        
    print('total frames: ' + str(cap_max))

    #release video
    cap.release() 
    
    #return motion values
    return(Motion)

########################################################################################

def Measure_Freezing(Motion,FreezeThresh,MinDuration):

    #Find frames below thresh
    BelowThresh = (Motion<FreezeThresh).astype(int)

    #Perform local cumulative thresh detection
    #For each consecutive frame motion is below threshold count is increased by 1 until motion goes above thresh, at which point coint is set back to 0
    CumThresh = np.zeros(len(Motion))
    for x in range (1,len(Motion)):
        if (BelowThresh[x]==1):
            CumThresh[x] = CumThresh[x-1] + BelowThresh[x]

    #Measure Freezing
    Freezing = (CumThresh>=MinDuration).astype(int) #whenever motion has dropped below the thresh for at least MinDuration call this freezing
    
    #the following code makes it so that the initial 30 before CumThresh meets MinDuration are still counted as freezing
    #Debating whether the code renders better results
    for x in range( len(Freezing) - 2, -1, -1) : 
        if Freezing[x] == 0 and Freezing[x+1]>0 and Freezing[x+1]<MinDuration:
            Freezing[x] = Freezing[x+1] + 1
    Freezing = (Freezing>0).astype(int)
    
    #Convert to Percentage
    Freezing = Freezing*100
    
    return(Freezing)

########################################################################################

def PlayVideo(fpath,fps,start,end,img_scale,save_video,Freezing,mt_cutoff,ycrop,SIGMA):
    
    #Upoad file
    cap = cv2.VideoCapture(fpath)
    
    #redfine start/end in frames
    fstart = start #*fps
    fend = end #*fps

    #set play speed and define first frame
    rate = int(1000/fps) #duration each frame is present for, in milliseconds
    cap.set(1,fstart) #set reference position of first frame to 0

    #set text parameters
    textfont = cv2.FONT_HERSHEY_SIMPLEX
    textposition = (10,30)
    textfontscale = 1
    textlinetype = 2

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_new = frame_new[ycrop:,:]
    frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA)

    #Initialize video storage if desired
    if save_video:
        width = int(frame.shape[1]*img_scale)
        height = int((frame.shape[0]+frame.shape[0]-ycrop)*img_scale)
        fourcc = cv2.VideoWriter_fourcc(*'jpeg') #only writes up to 20 fps, though video read can be 30.
        #fourcc = cv2.VideoWriter_fourcc(*'MJPG') #only writes up to 20 fps, though video read can be 30.
        writer = cv2.VideoWriter('out.avi', fourcc, 20.0, (width, height),isColor=False)

    #Loop through frames to detect frame by frame differences
    for x in range (fstart+1,fend):

        # press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Attempt to load next frame
        ret, frame = cap.read()
        if ret == True:

            #Convert to gray scale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #set old frame to new frame
            frame_old = frame_new

            #Reset new frame and process
            frame_new = frame
            frame_new = frame_new[ycrop:,:]
            frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA) # used to reduce influence of jitter from one frame to the next

            #Calculate difference between frames
            frame_dif = np.absolute(frame_new - frame_old)
            frame_cut = frame_dif > mt_cutoff
            frame_cut = frame_cut.astype('uint8')*255

            #Add text to videos
            if Freezing[x]==1:
                texttext = 'FREEZING'
                textfontcolor = 255
            else:
                texttext = 'ACTIVE'
                textfontcolor = 0
            cv2.putText(frame,texttext,textposition,textfont,textfontscale,textfontcolor,textlinetype)

            #Display video
            frame = cv2.resize(frame, (0,0), fx=img_scale, fy=img_scale) 
            frame_cut = cv2.resize(frame_cut, (0,0), fx=img_scale, fy=img_scale)
            preview = np.concatenate((frame,frame_cut))
            cv2.imshow("preview",preview)
            cv2.waitKey(rate)

            #Save video (if desired). 
            if save_video:
                writer.write(preview) 

        else: 
            print('No frame detected at frame : ' + str(x) + '.Stopping video play')
            break

    #Close video window and video writer if open        
    cv2.destroyAllWindows()
    _=cv2.waitKey(1) 
    if save_video:
        writer.release()
    
########################################################################################    
      
def SaveData(file,fpath,Motion,Freezing,fps,mt_cutoff,FreezeThresh,MinDuration):
    
    #Set output name
    fpath_out = fpath[:-4] + '_FreezingOutput.csv'

    #Create Dataframe
    DataFrame = pd.DataFrame(
        {'File': [file]*len(Motion),
         'FPS': np.ones(len(Motion))*fps,
         'MotionCutoff':np.ones(len(Motion))*mt_cutoff,
         'FreezeThresh':np.ones(len(Motion))*FreezeThresh,
         'MinFreezeDuration':np.ones(len(Motion))*MinDuration,
         'Frame': np.arange(len(Motion)),
         'Motion': Motion,
         'Freezing': Freezing
        })   

    DataFrame.to_csv(fpath_out)
    
########################################################################################        
    
def Summarize(file,Motion,Freezing,Bin_Names,Bin_Start,Bin_Stop,fps,mt_cutoff,FreezeThresh,MinDuration,Use_Bins):
    
    if Use_Bins == True:
        if len(Motion)<max(Bin_Start):
            print('Bin parameters exceed length of video.  Cannot create summary')
    elif Use_Bins == False:
        Bin_Names = ['avg'] 
        Bin_Start = [0] 
        Bin_Stop = [len(Motion)] 
    
    #Initialize arrays to store summary values in
    mt = np.zeros(len(Bin_Names)) 
    fz = np.zeros(len(Bin_Names))
    
    #Get averages for each bin
    for Bin in range (len(Bin_Names)):
        if len(Motion)<Bin_Stop[Bin]: #if end of video falls within bin, truncate bin to match video length
            mt[Bin]=np.mean(Motion[Bin_Start[Bin] : len(Motion)])
            fz[Bin]=np.mean(Freezing[Bin_Start[Bin] : len(Motion)])
        else:
            mt[Bin]=np.mean(Motion[Bin_Start[Bin] : Bin_Stop[Bin]])
            fz[Bin]=np.mean(Freezing[Bin_Start[Bin] : Bin_Stop[Bin]])
            
    #Create data frame to store data in
    df = pd.DataFrame(
    {'File': [file]*len(Bin_Names),
     'FileLength': np.ones(len(Bin_Names))*len(Motion),
     'FPS': np.ones(len(Bin_Names))*fps,
     'MotionCutoff':np.ones(len(Bin_Names))*mt_cutoff,
     'FreezeThresh':np.ones(len(Bin_Names))*FreezeThresh,
     'MinFreezeDuration':np.ones(len(Bin_Names))*MinDuration,
     'Bin': Bin_Names,
     'Bin_Start(f)': Bin_Start,
     'Bin_Stop(f)': Bin_Stop,
     'Motion': mt,
     'Freezing': fz
    })     
    return(df)

########################################################################################

def Batch(dpath,ftype,fps,ycrop,SIGMA,mt_cutoff,FreezeThresh,MinDuration,Use_Bins,Bin_Names,Bin_Start,Bin_Stop):
    
    #Convert necessary parameters from seconds to frames
    MinDuration = MinDuration * fps 
    Bin_Start = [x * fps for x in Bin_Start]
    Bin_Stop = [x * fps for x in Bin_Stop]

    #Get list of video files
    if os.path.isdir(dpath):
        FileNames = sorted(os.listdir(dpath))
        FileNames = fnmatch.filter(FileNames, ('*.' + ftype)) #restrict files to .mpg videos
    else:
        print('Directory not found. Check that path is correct.')

    #Loop through files    
    for file in FileNames:

        #Set file
        fpath = dpath + "/" + file
        print('Processing: ' + file)

        #Analyze frame by frame motion and freezing and save csv of results
        Motion = Measure_Motion(fpath,ycrop,mt_cutoff,SIGMA)
        Freezing = Measure_Freezing(Motion,FreezeThresh,MinDuration)
        SaveData(file,fpath,Motion,Freezing,fps,mt_cutoff,FreezeThresh,MinDuration)
        summary = Summarize(file,Motion,Freezing,Bin_Names,Bin_Start,Bin_Stop,fps,mt_cutoff,FreezeThresh,MinDuration,Use_Bins)

        #Add summary info for individual file to larger summary of all files
        try:
            summary_all = pd.concat([summary_all,summary])
        except NameError: #to be done for first file in list, before summary_all is created
            summary_all = summary

    #Write summary data to csv file
    sumpath_out = dpath + "/" + 'Summary.csv'
    summary_all.to_csv(sumpath_out)

########################################################################################

def Calibrate(fpath,cal_sec,cal_pix,fps,SIGMA):
    
    #Upoad file
    cap = cv2.VideoCapture(fpath)
    
    #set seconds to examine and frames
    cal_frames = cal_sec*fps

    #Initialize matrix for difference values
    cal_dif = np.zeros((cal_frames,cal_pix))

    #Initialize video
    cap.set(1,0) #first index references frame property, second specifies next frame to grab

    #Initialize first frame
    ret, frame = cap.read()
    frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA)

    #Get random set of pixels to examine across frames
    h,w=frame_new.shape
    h_loc = np.random.rand(cal_pix,1)*h
    h_loc = h_loc.astype(int)
    w_loc = np.random.rand(cal_pix,1)*w
    w_loc = w_loc.astype(int)

    #Loop through frames to detect frame by frame differences
    for x in range (1,cal_frames):

        #Reset old frame
        frame_old = frame_new

        #Load next frame
        ret, frame = cap.read()

        #Process frame
        frame_new = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_new = mh.gaussian_filter(frame_new,sigma=SIGMA) # used to reduce influence of jitter from one frame to the next

        #Get differences for select pixels
        frame_pix_dif = np.absolute(frame_new[h_loc,w_loc] - frame_old[h_loc,w_loc])
        frame_pix_dif = frame_pix_dif[:,0]

        #Populate difference array
        cal_dif[x,:]=frame_pix_dif

    #Attempt to remove differences due to movement of animal
    #if Rmv_Animal:
    #    otsu_rmv_animal = (mh.otsu((cal_dif*1000).astype('uint64')))/1000 #otsu will not work on non-integers.  This allows rounding to nearest 1000
    #    print('Otsu to remove animal: ' + str(otsu_rmv_animal))
    #    ninetynine_point_ninenine = np.percentile(cal_dif[cal_dif<otsu_rmv_animal],99.99)
    #else:
    #    ninetynine_point_ninenine = np.percentile(cal_dif,99.99)      
    
    ninetynine_point_ninenine = np.percentile(cal_dif,99.99)

    #Calculate grayscale change cutoff for detecting motion
    cal_dif_avg = np.nanmean(cal_dif)

    #Set Cutoff
    mt_cutoff = 2*ninetynine_point_ninenine

    #Print stats and selected cutoff
    print ('Average frame-by-frame pixel difference: ' + str(cal_dif_avg))
    print ('99.99 percentile of pixel change differences: ' + str(ninetynine_point_ninenine))
    print ('Grayscale change cut-off for pixel change: ' + str(mt_cutoff))


