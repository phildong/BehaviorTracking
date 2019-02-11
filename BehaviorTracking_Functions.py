#
#List of Functions in FreezeAnalysis_Functions
#

# Check -
# LoadAndCrop - 
# Load_First -
# Measure_Motion -
# Measure_Freezing -
# Play_Video -
# Save_Data -
# Summarize -
# Batch -
# Calibrate -
# Reference -
# Locate -
# TrackLocation -
# LocationThresh_View -
# ROI_plot -
# ROI_Location -

########################################################################################

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
from scipy import ndimage
import holoviews as hv
from holoviews import opts
from holoviews import streams
from holoviews.streams import Stream, param
hv.notebook_extension('bokeh')
warnings.filterwarnings("ignore")

########################################################################################

def Check(Bin_Names,Bin_Start,Bin_Stop):
    if len(Bin_Names)!=len(Bin_Start)!=len(Bin_Stop):
        print('WARNING.  Bin list sizes are not of equal length')  

########################################################################################        

def LoadAndCrop(dpath,file,stretch_w=1,stretch_h=1,cropmethod='none'):
    
    #Upoad file and check that it exists
    fpath = dpath + "/" + file   
    if os.path.isfile(fpath):
        print('file: '+ fpath)
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')

    #Get maxiumum frame of file. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(7)) #7 is index of total frames
    print('total frames: ' + str(cap_max))

    #Retrieve first frame
    cap.set(1,0) #first index references frame property, second specifies next frame to grab
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    cap.release() 

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(gray.shape[1]), np.arange(gray.shape[0]), gray))
    image.opts(width=int(gray.shape[1]*stretch_w),
               height=int(gray.shape[0]*stretch_h),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='above',
              title="First Frame.  Crop if Desired")
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod=='none':
        image.opts(title="First Frame")
        return image,None,fpath
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        box_stream = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box),box_stream,fpath
    
    if cropmethod=='HLine':  
        points = hv.Points([])
        points.opts(active_tools=['point_draw'], color='white',size=1)
        pointerXY_stream = streams.PointerXY(x=0, y=0, source=image)
        pointDraw_stream = streams.PointDraw(source=points,num_objects=1)
            
        def h_track(x, y): #function to track pointer
            y = int(np.around(y))
            text = hv.Text(x, y, str(y), halign='left', valign='bottom')
            return hv.HLine(y) * text
        track=hv.DynamicMap(h_track, streams=[pointerXY_stream])
        
        def h_line(data): #function to draw line
            try:
                hline=hv.HLine(data['y'][0])
                return hline
            except:
                hline=hv.HLine(0)
                return hline
        line=hv.DynamicMap(h_line,streams=[pointDraw_stream])
        
        def h_text(data): #function to write ycrop value
            center=gray.shape[1]//2 
            try:
                y=int(np.around(data['y'][0]))
                htext=hv.Text(center,y+5,'ycrop: {x}'.format(x=y))
                return htext
            except:
                htext=hv.Text(center,5, 'ycrop: 0')
                return htext
        text=hv.DynamicMap(h_text,streams=[pointDraw_stream])
        
        return image*track*points*line*text,pointDraw_stream,fpath    
    
########################################################################################    
    
def Load_First(dpath,file):
    
    #Upoad file
    fpath = dpath + "/" + file
    
    if os.path.isfile(fpath):
        print('file: '+ fpath)
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')

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
    #For each consecutive frame motion is below threshold count is increased by 1 until motion goes above thresh, 
    #at which point coint is set back to 0
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
            if Freezing[x]==100:
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
    
    Bin_Start = [x * fps for x in Bin_Start]
    Bin_Stop = [x * fps for x in Bin_Stop]
    
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

    percentile = np.percentile(cal_dif,99.99)

    #Calculate grayscale change cutoff for detecting motion
    cal_dif_avg = np.nanmean(cal_dif)

    #Set Cutoff
    mt_cutoff = 2*percentile

    #Print stats and selected cutoff
    print ('Average frame-by-frame pixel difference: ' + str(cal_dif_avg))
    print ('99.99 percentile of pixel change differences: ' + str(percentile))
    print ('Grayscale change cut-off for pixel change: ' + str(mt_cutoff))
    
    hist_freqs, hist_edges = np.histogram(cal_dif,bins=np.arange(50))
    hist = hv.Histogram((hist_edges, hist_freqs))
    hist.opts(title="Motion Cutoff",xlabel="Grayscale Change")
    vline = hv.VLine(mt_cutoff)
    vline.opts(color='red')
    return hist*vline
    

########################################################################################
    
def Reference(fpath,crop,f):

    #Upoad file
    cap = cv2.VideoCapture(fpath)
    cap.set(1,0)#first index references frame property, second specifies next frame to grab
    
    #Get video dimensions with any cropping applied
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
    except:
        fxmin,fxmax=0,frame.shape[1]
        fymin,fymax=0,frame.shape[0]
    h,w=(fymax-fymin),(fxmax-fxmin)
    cap_max = int(cap.get(7)) #7 is index of total frames
    
    #Collect subset of frames
    collection = np.zeros((f,h,w))  
    for x in range (f):          
        grabbed = False
        while grabbed == False: 
            y=np.random.randint(0,cap_max)
            cap.set(1,y)#first index references frame property, second specifies next frame to grab
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray[fymin:fymax,fxmin:fxmax]
                collection[x,:,:]=gray
                grabbed = True
            elif ret == False:
                pass
    cap.release() 
    
    reference = np.median(collection,axis=0)
    return reference    

########################################################################################

def Locate(f,reference,SIGMA,loc_thresh,cap,crop,use_window=False,window=None,window_weight=0,prior=None):
    
    #attempt to load frame
    cap.set(1,f) #first index references frame property, second specifies next frame to grab
    ret, frame = cap.read() #read frame
    
    #set window dimensions
    if window != None:
        window = window//2
        ymin,ymax = prior[0]-window, prior[0]+window
        xmin,xmax = prior[1]-window, prior[1]+window

    if ret == True:
        
        #load frame and crop
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            Xs=[crop.data['x0'][0],crop.data['x1'][0]]
            Ys=[crop.data['y0'][0],crop.data['y1'][0]]
            fxmin,fxmax=int(min(Xs)), int(max(Xs))
            fymin,fymax=int(min(Ys)), int(max(Ys))
        except:
            fxmin,fxmax=0,frame.shape[1]
            fymin,fymax=0,frame.shape[0]
        frame = frame[fymin:fymax,fxmin:fxmax]
        
        #find difference from reference and blur
        dif = np.absolute(frame-reference)
        dif = mh.gaussian_filter(dif,sigma=SIGMA)
        
        #apply window
        if (use_window==True) and (window != None):
            dif_weights = np.ones(dif.shape)*window_weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,loc_thresh)]=0
        com=ndimage.measurements.center_of_mass(dif)
        return ret, dif, com
    
    else:
        return ret, None, None
        
########################################################################################        

def TrackLocation(fpath,reference,SIGMA,loc_thresh,crop,use_window=False,window=None,window_weight=0):
    
    print('use_window: {}'.format(use_window))
    print('window: {}'.format(window))
    print('window_weight: {}'.format(window_weight))
    
    
    #load video
    cap = cv2.VideoCapture(fpath)#set file
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max)
    Y = np.zeros(cap_max)
    D = np.zeros(cap_max)

    #Loop through frames to detect frame by frame differences
    for f in range (cap_max):
        
        if f>0: 
            yprior = np.around(Y[f-1]).astype(int)
            xprior = np.around(X[f-1]).astype(int)
            ret,dif,com = Locate(f,reference,SIGMA,loc_thresh,cap,crop,use_window,window,window_weight,prior=[yprior,xprior])
        else:
            ret,dif,com = Locate(f,reference,SIGMA,loc_thresh,cap,crop)
                                                
        if ret == True:          
            Y[f] = com[0]
            X[f] = com[1]
            if f>0:
                D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
        else:
            #if no frame is detected
            cap_max = (f-1) #Reset max frame to last frame detected
            X = X[:cap_max] #Amend length of X vector
            Y = Y[:cap_max] #Amend length of Y vector
            D = D[:cap_max] #Amend length of D vector
            break   
    
    #release video
    cap.release()
    print('total frames: ' + str(cap_max))
    
    #return pandas dataframe
    df = pd.DataFrame(
    {'Frame': np.arange(len(X)),
     'X': X,
     'Y': Y,
     'Distance': D
    })    
    return df

########################################################################################

def LocationThresh_View(examples,figsize,fpath,reference,SIGMA,loc_thresh,crop):
    
    #load video
    plt.figure(figsize=figsize)
    cap = cv2.VideoCapture(fpath)#set file
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    
    #define cropping values
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
    except:
        fxmin,fxmax=0,frame.shape[1]
        fymin,fymax=0,frame.shape[0]
    
    #examine random frames
    for x in range (1,examples+1):
        
        #analyze frame
        f=np.random.randint(0,cap_max) #select random frame
        cap.set(1,f) #sets frame to be next to be grabbed
        ret,dif,com = Locate(f,reference,SIGMA,loc_thresh,cap,crop) #get frame difference from reference 

        #plot original frame
        plt.subplot(2,examples,x)
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame = frame[fymin:fymax,fxmin:fxmax]
        plt.annotate('COM', 
                     xy=(com[1], com[0]), xytext=(com[1]-20, com[0]-20),
                     color='red',
                     size=15,
                     arrowprops=dict(facecolor='red'))
        plt.title('Frame ' + str(f))
        plt.imshow(frame)
        plt.gray()
        
        #plot difference
        plt.subplot(2,examples,(x+examples))
        plt.annotate('COM',
                     xy=(com[1], com[0]), xytext=(com[1]-20, com[0]-20),
                     color='white',
                     size=15,
                     arrowprops=dict(facecolor='white'))
        plt.imshow(dif)
        plt.jet()
    
    #release cap when done
    cap.release()

########################################################################################    
    
def ROI_plot(reference,region_names,stretch_w=1,stretch_h=1):
    
    #Define parameters for plot presentation
    nobjects = len(region_names) #get number of objects to be drawn

    #Make reference image the base image on which to draw
    image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference))
    image.opts(width=int(reference.shape[1]*stretch_w),
               height=int(reference.shape[0]*stretch_h),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='above',
              title="Draw Regions: "+', '.join(region_names))

    #Create polygon element on which to draw and connect via stream to PolyDraw drawing tool
    poly = hv.Polygons([])
    poly_stream = streams.PolyDraw(source=poly, drag=True, num_objects=nobjects, show_vertices=True)
    poly.opts(fill_alpha=0.3, active_tools=['poly_draw'])

    def centers(data):
        try:
            x_ls, y_ls = data['xs'], data['ys']
        except TypeError:
            x_ls, y_ls = [], []
        xs = [np.mean(x) for x in x_ls]
        ys = [np.mean(y) for y in y_ls]
        rois = region_names[:len(xs)]
        return hv.Labels((xs, ys, rois))

    dmap = hv.DynamicMap(centers, streams=[poly_stream])
    
    return (image * poly * dmap), poly_stream

########################################################################################    

def ROI_Location(reference,poly_stream,region_names,location):

    #Create ROI Masks
    ROI_masks = {}
    for poly in range(len(poly_stream.data['xs'])):
        x = np.array(poly_stream.data['xs'][poly]) #x coordinates
        y = np.array(poly_stream.data['ys'][poly]) #y coordinates
        xy = np.column_stack((x,y)).astype('uint64') #xy coordinate pairs
        mask = np.zeros(reference.shape) # create empty mask
        cv2.fillPoly(mask, pts =[xy], color=255) #fill polygon  
        ROI_masks[region_names[poly]] = mask==255 #save to ROI masks as boolean 

    #Create arrays to store whether animal is within given ROI
    ROI_location = {}
    for mask in ROI_masks:
        ROI_location[mask]=np.full(len(location['Frame']),False,dtype=bool)

    #For each frame assess truth of animal being in each ROI
    for f in location['Frame']:
        y,x = location['Y'][f], location['X'][f]
        for mask in ROI_masks:
            ROI_location[mask][f] = ROI_masks[mask][int(y),int(x)]
    
    #Add date to location data frame
    for x in ROI_location:
        location[x]=ROI_location[x]
    
    return location
    
    
    
    
    
    
    
    
    
