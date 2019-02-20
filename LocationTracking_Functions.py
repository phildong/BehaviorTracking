#
#List of Functions in LocationTracking_Functions.py
#

# Check -
# LoadAndCrop - 
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

def Check(UseBins,bin_dict):
    
    if UseBins==True:
        if len(bin_dict['Bin_Names']) != len(bin_dict['Bin_Start']) or len(bin_dict['Bin_Names']) != len(bin_dict['Bin_Stop']):
            print('WARNING.  Bin list sizes are not of equal length') 

########################################################################################        

def LoadAndCrop(video_dict,stretch,cropmethod='none'):
    
    #Upoad file and check that it exists
    video_dict['fpath'] = os.path.join(os.path.normpath(video_dict['dpath']), video_dict['file'])
    if os.path.isfile(video_dict['fpath']):
        print('file: {file}'.format(file=video_dict['fpath']))
        cap = cv2.VideoCapture(video_dict['fpath'])
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')

    #Get maxiumum frame of file. Note that max frame is updated later if fewer frames detected
    cap_max = int(cap.get(7)) #7 is index of total frames
    print('total frames: {frames}'.format(frames=cap_max))

    #Set first frame
    cap.set(1,video_dict['start']) #first index references frame property, second specifies next frame to grab
    ret, frame = cap.read() 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)   
    cap.release() 

    #Make first image reference frame on which cropping can be performed
    image = hv.Image((np.arange(frame.shape[1]), np.arange(frame.shape[0]), frame))
    image.opts(width=int(frame.shape[1]*stretch['width']),
               height=int(frame.shape[0]*stretch['height']),
              invert_yaxis=True,cmap='gray',
              colorbar=True,
               toolbar='below',
              title="First Frame.  Crop if Desired")
    
    #Create polygon element on which to draw and connect via stream to poly drawing tool
    if cropmethod=='none':
        image.opts(title="First Frame")
        return image,None,video_dict
    
    if cropmethod=='Box':         
        box = hv.Polygons([])
        box.opts(alpha=.5)
        box_stream = streams.BoxEdit(source=box,num_objects=1)     
        return (image*box),box_stream,video_dict
    
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
            center=frame.shape[1]//2 
            try:
                y=int(np.around(data['y'][0]))
                htext=hv.Text(center,y+10,'ycrop: {x}'.format(x=y))
                return htext
            except:
                htext=hv.Text(center,10, 'ycrop: 0')
                return htext
        text=hv.DynamicMap(h_text,streams=[pointDraw_stream])
        
        return image*track*points*line*text,pointDraw_stream,video_dict   
    
    

########################################################################################
    
def Reference(video_dict,crop,num_frames=100):
    
    #get correct ref video
    vname = video_dict.get("altfile", video_dict['file'])
    fpath = os.path.join(os.path.normpath(video_dict['dpath']), vname)
    if os.path.isfile(fpath):
        print('file: {file}'.format(file=fpath))
        cap = cv2.VideoCapture(fpath)
    else:
        raise FileNotFoundError('File not found. Check that directory and file names are correct.')

    #Upoad file
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
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #Collect subset of frames
    collection = np.zeros((num_frames,h,w))  
    for x in range (num_frames):          
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

def Locate(cap,crop,reference,tracking_params,prior=None):    
    
    #attempt to load frame
    ret, frame = cap.read() #read frame
    
    #set window dimensions
    if prior != None and tracking_params['use_window']==True:
        window_size = tracking_params['window_size']//2
        ymin,ymax = prior[0]-window_size, prior[0]+window_size
        xmin,xmax = prior[1]-window_size, prior[1]+window_size

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
        dif = dif.astype('uint8')
              
        #apply window
        weight = 1 - tracking_params['window_weight']
        if prior != None and tracking_params['use_window']==True:
            dif_weights = np.ones(dif.shape)*weight
            dif_weights[slice(ymin if ymin>0 else 0, ymax),
                        slice(xmin if xmin>0 else 0, xmax)]=1
            dif = dif*dif_weights
            
        #threshold differences and find center of mass for remaining values
        dif[dif<np.percentile(dif,tracking_params['loc_thresh'])]=0
        com=ndimage.measurements.center_of_mass(dif)
        return ret, dif, com
    
    else:
        return ret, None, None
        
########################################################################################        

def TrackLocation(video_dict,tracking_params,reference,crop):
    
    print('use_window: {}'.format(tracking_params['use_window']))
    print('window_size: {}'.format(tracking_params['window_size']))
    print('window_weight: {}'.format(tracking_params['window_weight']))
    
    
    #load video
    cap = cv2.VideoCapture(video_dict['fpath'])#set file
    cap.set(1,video_dict['start']) #set starting frame
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
   
    
    #Initialize vector to store motion values in
    X = np.zeros(cap_max - video_dict['start'])
    Y = np.zeros(cap_max - video_dict['start'])
    D = np.zeros(cap_max - video_dict['start'])

    #Loop through frames to detect frame by frame differences
    for f in range(len(D)):
        
        if f>0: 
            yprior = np.around(Y[f-1]).astype(int)
            xprior = np.around(X[f-1]).astype(int)
            ret,dif,com = Locate(cap,crop,reference,tracking_params,prior=[yprior,xprior])
        else:
            ret,dif,com = Locate(cap,crop,reference,tracking_params)
                                                
        if ret == True:          
            Y[f] = com[0]
            X[f] = com[1]
            if f>0:
                D[f] = np.sqrt((Y[f]-Y[f-1])**2 + (X[f]-X[f-1])**2)
        else:
            #if no frame is detected
            f = f-1
            X = X[:f] #Amend length of X vector
            Y = Y[:f] #Amend length of Y vector
            D = D[:f] #Amend length of D vector
            break   
    
    #release video
    cap.release()
    print('total frames processed: {f}'.format(f=len(D)))
    
    #create pandas dataframe
    df = pd.DataFrame(
    {'File' : video_dict['file'],
     'FPS': np.ones(len(D))*video_dict['fps'],
     'Location_Thresh': np.ones(len(D))*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(len(D))*tracking_params['window_weight'],
     'Window_Size': np.ones(len(D))*tracking_params['window_size'],
     'Start_Frame': np.ones(len(D))*video_dict['start'],
     'Frame': np.arange(len(D)),
     'X': X,
     'Y': Y,
     'Distance': D
    })
    

    
    return df


########################################################################################

def LocationThresh_View(examples,figsize,video_dict,reference,crop,tracking_params):
    
    #load video
    plt.figure(figsize=figsize)
    cap = cv2.VideoCapture(video_dict['fpath'])
    cap_max = int(cap.get(7)) #get max frames. 7 is index of total frames
    cap_max = int(video_dict['end']) if video_dict['end'] is not None else cap_max
    
    #define cropping values
    try:
        Xs=[crop.data['x0'][0],crop.data['x1'][0]]
        Ys=[crop.data['y0'][0],crop.data['y1'][0]]
        fxmin,fxmax=int(min(Xs)), int(max(Xs))
        fymin,fymax=int(min(Ys)), int(max(Ys))
    except:
        fxmin,fxmax=0,reference.shape[1]
        fymin,fymax=0,reference.shape[0]
    
    #examine random frames
    for x in range (1,examples+1):
        
        #analyze frame
        f=np.random.randint(0,cap_max) #select random frame
        cap.set(1,f) #sets frame to be next to be grabbed
        ret,dif,com = Locate(cap,crop,reference,tracking_params) #get frame difference from reference 

        #plot original frame
        plt.subplot(2,examples,x)
        cap.set(1,f) #resets frame position
        ret, frame = cap.read() #read frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        frame = frame[fymin:fymax,fxmin:fxmax]
        plt.annotate('COM', 
                     xy=(com[1], com[0]), xytext=(com[1]-20, com[0]-20),
                     color='red',
                     size=15,
                     arrowprops=dict(facecolor='red'))
        plt.title('Frame: {f}'.format(f=f))
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
    
def ROI_plot(reference,region_names,stretch):
    
    #Define parameters for plot presentation
    nobjects = len(region_names) #get number of objects to be drawn

    #Make reference image the base image on which to draw
    image = hv.Image((np.arange(reference.shape[1]), np.arange(reference.shape[0]), reference))
    image.opts(width=int(reference.shape[1]*stretch['width']),
               height=int(reference.shape[0]*stretch['height']),
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
    
########################################################################################        
    
def Summarize_Location(video_dict,tracking_params,bin_dict,location,UseBins,region_names):  
   
    
    #redefine bins in terms of frames 
    Bin_Names,Bin_Start,Bin_Stop = bin_dict['Bin_Names'],bin_dict['Bin_Start'],bin_dict['Bin_Stop']
    if UseBins == True:
        Bin_Start = [x * video_dict['fps'] for x in Bin_Start]
        Bin_Stop = [x * video_dict['fps'] for x in Bin_Stop]
        if len(location)<max(Bin_Start):
            print('Bin parameters exceed length of video.  Some bin info will not be generated')
    elif UseBins == False:
        Bin_Names = ['avg'] 
        Bin_Start = [0] 
        Bin_Stop = [len(location)] 
    
    #initialize dataframe to store summary data in
    try:
        df = pd.DataFrame({var_name: np.zeros(len(Bin_Names)) for var_name in region_names})
        df['Distance']=np.zeros(len(Bin_Names))   
    except: #when no regions have been defined
        df = pd.DataFrame({'Distance':np.zeros(len(Bin_Names))})
    
    #calculate sum of motion and proportion of tme for each ROI
    for Bin in range(len(Bin_Names)):
        segment = slice(Bin_Start[Bin],Bin_Stop[Bin])
        try:
            df.loc[Bin] = location.loc[segment].apply(
                dict([('Distance', np.sum)] + [(rname, np.mean) for rname in region_names]))  
        except: #when no regions have been defined
            df.loc[Bin] = location.loc[segment].apply(dict([('Distance', np.sum)]))  
            
    #Create data frame to store data in
    length = len(Bin_Names)
    df_summary = pd.DataFrame(
    {'File': video_dict['file']*length,
     'FileLength': np.ones(length)*len(location),
     #'FPS': np.ones(length)*video_dict['fps'],
     'Location_Thresh': np.ones(length)*tracking_params['loc_thresh'],
     'Use_Window': str(tracking_params['use_window']),
     'Window_Weight': np.ones(length)*tracking_params['window_weight'],
     'Window_Size': np.ones(length)*tracking_params['window_size'],
     'Bin': Bin_Names,
     'Bin_Start(f)': Bin_Start,
     'Bin_Stop(f)': Bin_Stop,
    })    
    df_summary = pd.concat([df_summary,df],axis=1)
    return(df_summary) 
    
########################################################################################        
#Code to export svg
#conda install -c conda-forge selenium phantomjs

#import os
#from bokeh import models
#from bokeh.io import export_svgs

#bokeh_obj = hv.renderer('bokeh').get_plot(image).state
#bokeh_obj.output_backend = 'svg'
#export_svgs(bokeh_obj, dpath + '/' + 'Calibration_Frame.svg')
    