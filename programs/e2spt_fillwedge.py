#!/usr/bin/env python

# Author: Jesus Galaz-Montoya, 15/August/2014, last update 18/August/2014
# Copyright (c) 2011 Baylor College of Medicine
#
# This software is issued under a joint BSD/GNU license. You may use the
# source code in this file under either license. However, note that the
# complete EMAN2 and SPARX software packages have some GPL dependencies,
# so you are responsible for compliance with the licenses of these packages
# if you opt to use BSD licensing. The warranty disclaimer below holds
# in either instance.
#
# This complete copyright notice must be included in any revised version of the
# source code. Additional authorship citations may be added, but existing
# author citations must be preserved.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  2111-1307 USA


import os
from EMAN2 import *
import sys
import math


def main():
	progname = os.path.basename(sys.argv[0])
	usage = """UNDER DEVELOPOMENT. Uses the supplied tilt series (in HDF format) to reconstruct a 3-D volume. To convert a tilt series to an HDF stack use e2spt_tiltstacker.py. Then, projections are collected from angular directions that were not experimentally available. A new 3-D volume is reconstructed using the existing data, and the new generated projections."""

	parser = EMArgumentParser(usage=usage,version=EMANVERSION)
	
	parser.add_argument('--input',type=str,default='',help="""Comma separated files in .ali, .st .hdf format of the aligned tiltseries.""")
	
	parser.add_argument('--inputstem',type=str,default='',help="""Alternative to supplying --input. This is a string common to multiple files to be processed in the CURERENT directory. The common string doesn't need to be at a particular location in the filenames. For example, a series of files "tiltA.hdf, tiltB.hdf, tiltC.hdf" could have either 'hdf', '.hdf', 't,','ti', 'til', 'tilt', etc., as a common string. The key is to choose a string shared ONLY by the files of interest. The files should be multiple subtiltseries in .hdf format; each file should correspond to an individual subtiltseries for a different particle: That is, each file should be a subtiltseries corresponding to an individual subtomogram, as extracted by e2spt_subtilt.py, or as simulated by e2spt_simulation.py""")

	parser.add_argument('--inputdir',type=str,default='',help="""Alternative to --input and --inputstem. Path to a directory containing individual subtiltseries stacks.""")
	
	parser.add_argument("--iter", type=int, help="""The number of iterations to perform. Default is 1.""", default=1)
		
	parser.add_argument('--path',type=str,default='fillwedge',help="""Directory to save the results.""")
	
	#parser.add_argument("--finalboxsize", type=int,default=0,help="""The final box size 
	#	to clip the final 3D reconstruction to.""")	
	
	#parser.add_argument('--subset', type=int, default=0, help='''Specify how many sub-tiltseries 
	#	(or particles) from the coordinates file you want to extract; e.g, if you specify 10, 
	#	the first 10 particles will be boxed.\n0 means "box them all" because it makes no 
	#	sense to box none''')

	parser.add_argument("--reconstructor", type=str,default="fourier",help="""The reconstructor to use to reconstruct the tilt series into a tomogram. Type 'e2help.py reconstructors' at the command line to see all options and parameters available. To specify the interpolation scheme for the fourier reconstruction, specify 'mode'. Options are 'nearest_neighbor', 'gauss_2', 'gauss_3', 'gauss_5', 'gauss_5_slow', 'gypergeom_5', 'experimental'. For example --reconstructor=fourier:mode=gauss_5 """)
	
	parser.add_argument("--tiltaxis",type=str,default='y',help="""Axis to produce projections about. Default is 'y'; the only other valid option is 'x'.""")
		
	parser.add_argument("--tiltstep",type=int,default=0,help="""Tilt step to use to generate projections to fill in the missing wedge.""")
	
	parser.add_argument("--pad2d", type=float,default=0.0,help="""Padding factor to zero-pad the 2d images in the tilt series prior to reconstruction. (The final reconstructed subvolumes will be cropped to the original size).""")

	parser.add_argument("--pad3d", type=float,default=0.0,help="""Padding factor to zero-pad the reconstruction volume. (The final reconstructed subvolumes will be cropped to the original size).""")	
	
	parser.add_argument("--savevols",action='store_true',default=False,help="""This option will save the reconstructed volume before and after filling the missing wedge.""")
		
	parser.add_argument("--outxsize",type=int,default=0,help='''Clip the output volume in x to this size. The default size is the nx size of the input images.''')
	
	parser.add_argument("--outysize",type=int,default=0,help='''Clip the output volume in y to this size. The default size is the ny size of the input images.''')
	
	parser.add_argument("--outzsize",type=int,default=0,help='''Clip the output volume in z to this size. The default size is the nx size of the input images.''')
	
	parser.add_argument("--tltfile",type=str,default='',help="""IMOD-like .tlt file with tilt angles for the aligned tiltseries (or set of subtiltseries).""")
		
	parser.add_argument("--mask",type=str,help="""Masking processor (see e2help.py --verbose=10) applied to each volume prior to radial density plot computation. Default=None.""", default='')
	
	parser.add_argument("--preprocess",type=str,help="""Any processor (see e2help.py --verbose=10) applied to each volume prior to reprojection generation.""", default='')
	
	parser.add_argument("--lowpass",type=str,help="""Default=None. A lowpass filtering processor (see e2help.py --verbose=10) applied to each volume prior to reprojection generation.""", default='')
	
	parser.add_argument("--highpass",type=str,help="""Default=None. A highpass filtering processor (see e2help.py --verbose=10) applied to each volume prior to reprojection generation.""", default='')	
		
	parser.add_argument("--threshold",type=str,help="""Default=None. A threshold  processor (see e2help.py --verbose=10) applied to each volume prior to reprojection generation.""", default='')
		
	parser.add_argument("--maskfilling",type=str,help="""Masking processor (see e2help.py --verbose=10) applied to reprojections outside the tiltseries range, which will fill the missing wedge.""", default='')
	
	parser.add_argument("--preprocessfilling",type=str,help="""Any processor (see e2help.py --verbose=10) applied to reprojections outside the tiltseries range, which will fill the missing wedge.""", default='')
	
	parser.add_argument("--lowpassfilling",type=str,help="""Default=None. A lowpass filtering processor (see e2help.py --verbose=10) applied to reprojections outside the tiltseries range, which will fill the missing wedge.""", default='')
	
	parser.add_argument("--highpassfilling",type=str,help="""Default=None. A highpass filtering processor (see e2help.py --verbose=10) applied to reprojections outside the tiltseries range, which will fill the missing wedge.""", default='')	
		
	parser.add_argument("--thresholdfilling",type=str,help="""Default=None. A threshold  processor (see e2help.py --verbose=10) applied to reprojections outside the tiltseries range, which will fill the missing wedge.""", default='')
	
	parser.add_argument("--preproc2d",action='store_true',default=False,help="""Preprocessors such as --lowpass, --highpass, --threshold, etc., will be applied to all 2D projections instead of the reconstructed volume.""")
		
	parser.add_argument("--weightbyangle",action='store_true',default=False,help="""Default-False. Reprojections outside of the experimental data range will be weighed by the cosine of the angle, and by the iteration, such that they will reach a weight of 1 when there are as many iterations as images in the subtiltseries.""")
	
	parser.add_argument("--normalizetomiddleslice",action='store_true',default=False,help="""Applies the 'normalize.toimage' processor to the reprojections to fill in the missing wedge to match the middle slice (See e2help.py processors).""")
	
	#parser.add_argument("--normproc",type=str,help="""Not used anywhere yet. Default=None""", default='None')
	
	#parser.add_argument("--thresh",type=str,help="""Not used anywhere yet. Default=None""", default='None')

	parser.add_argument("--ppid", type=int, help="""Set the PID of the parent process, used for cross platform PPID""",default=-1)
	
	parser.add_argument("--verbose", "-v", dest="verbose", action="store", metavar="n",type=int, default=0, help="verbose level [0-9], higner number means higher level of verboseness")
	
	(options, args) = parser.parse_args()
	
	logger = E2init(sys.argv, options.ppid)
	
	from e2spt_classaverage import sptOptionsParser
	options = sptOptionsParser( options )
	
	#if options.reconstructor == 'None' or options.reconstructor == 'none':
	#	options.reconstructor = None
	
	#if options.reconstructor and options.reconstructor != 'None' and options.reconstructor != 'none': 
	
	
	#print "the reconstructor is", options.reconstructor
	
	#options.maskfilling=parsemodopt(options.maskfilling)
	
	from e2spt_classaverage import sptmakepath
	
	options = sptmakepath(options,'fillwedge')
	originalpath = options.path
	
	if options.verbose > 9:
		print "\n(e2spt_fillwedge.py) I've read the options"	
	
	inputfiles = {}											#C:Figure out whether there's a single HDF stack to process,
															#C:or a directory with many HDF stacks
	#if options.inputstem:
	c = os.getcwd()

	if options.inputdir:
		c = os.getcwd() + '/' + options.inputdir

	findir = os.listdir( c )
	
	if options.inputstem:
		for f in findir:
			if '.hdf' in f and options.inputstem in f:
				if options.verbose > 8:
					print "\nFound tiltseries!", f
				inputfiles.update( {f:[f,None]} )			#C:The input files are put into a dictionary in the format {originalseriesfile:[originalseriesfile,volumefile]}
	
	elif options.inputdir:
		for f in findir:
			if '.hdf' in f:
				if options.verbose > 8:
					print "\nFound tiltseries!", f
				inputfiles.update( {f:[f,None]} )			#C:The input files are put into a dictionary in the format {originalseriesfile:[originalseriesfile,volumefile]}
	
	elif options.input:
		fs=options.input.split(',')
		for f in fs:
			inputfiles.update( {f:[f,None]} )	
	
	originalangles = lowerangles = upperangles = allangles = 0
	tiltstep = 0
	vol = None
	newfiles = {}
	firstiterdir = originalpath
	for i in range( options.iter ):		#Iterate over options.iter
		previouspath=''
		if options.iter > 1:			
			iterdir = 'iter_' + str( i+1 ).zfill( len ( str( options.iter ) ) )
			os.system( 'mkdir ' + originalpath + '/' + iterdir )	#C:Make a new subdirectory within options.path only if options.iter > 0
			options.path =  originalpath + '/' + iterdir			#C:Update path to include the newly created subdirectory
			previouspath = originalpath + '/iter_' + str( i ).zfill( len ( str( options.iter ) ) )
			if i == 0:
				firstiterdir = options.path
		kk = 0
		
		
		for f in inputfiles:										#C:Iterate over files
			originalseries = f 
			
			if i == 0:
				if options.tltfile:
					originalangles = getangles( options )				#C:Read angles from tlt file if available
				else:
					originalangles = calcangles( originalseries )		#C:Get the angles of the actual data tilts from the header or from input parameters
		
			#originalangles = []
			currentseries = inputfiles[f][0]
			curentvolume = inputfiles[f][1]
			
			stackfile = options.path + '/' + os.path.basename(f).replace('.hdf','_WF3D.hdf')
			
			print "\nWill fill wedge for file", f
			if i ==0:
				hdr = EMData( f, 0, True )				#See if angles are in the header of the data, for each file; if not, write them
				print "\nRead header and its type", hdr, type(hdr)
				print "\nAnd the dictionary", hdr.get_attr_dict()
				
				aux=0
				if 'spt_tiltangle' not in hdr.get_attr_dict():
					print "\nspt_tiltangle not in header, therefore will write it by calling writeanglestoheader"
					aux = writeanglestoheader( f, originalangles )
					print "\naux returned is", aux
				else:
					aux = 1			 
				
				if aux:		
					retm = makevol ( options, originalseries, i, originalangles[0], originalangles[-1], 1 )	#C:In the first iteration, first reconstruct the tilt series into a 3D volume
					vol = retm[0]
					print "\nVol and its type are", vol,type(vol)
				
					retc = completeangles( options, originalangles )			#C:Get the angles needed to complete the missing wedge region
					lowerangles = retc[0]
					upperangles = retc[1]
					allangles = retc[2]
					tiltstep = retc[3]
					
					writeanglestofile( options, allangles )
				else:
					print "ERROR: Something went wrong. spt_tiltangle found in image headers, but somehow is unusuable"
					sys.exit()
					
			elif i>0:
				#vol = EMData( inputfiles[f][1] )	#C:For iterations > 0 (after the first iteration), the updated volumes 
													#C:for each tilt series should exist from the previous iteration						
				
				previousstackfile =previouspath + '/' + os.path.basename(f).replace('.hdf','_WF3D.hdf')

				#previousstackfile = previouspath + '/stack3D.hdf'
				print "\n\n\previousstackfile is", previousstackfile
				vol = EMData( previousstackfile, 0 )
				#C:Make projections for all missing tilts with which to fill the missing wedge
			
			retw = wedgefiller( options, originalseries, vol, originalangles, lowerangles, upperangles, allangles, tiltstep, i )		#Generate projections to fill in the missing wedge
			newseries = retw[ 0 ]														
			newvol = retw[ 1 ]
			newvolfile = retw[ 2 ]
			
			if i == 0:
				print "\nEEEEEEEEEEEEEEE\n\n\n\n\nNewvolfile returned is", newvolfile
		
			newfiles.update( {originalseries:[newseries,newvolfile]} )
			
			newvol.write_image( stackfile, 0 )
			
			kk+=1				
			
			#firstrawvolfile = firstiterdir + '/' + os.path.basename(f).replace('.hdf','_3D.hdf')
			#fscfile = options.path + '/' + os.path.basename(f).replace('.hdf','FSC.txt')
			#cmdfsc = 'e2proc3d.py ' + firstrawvolfile + ' ' + fscfile + ' --calcfsc=' + newvolfile 
			#os.popen( cmdfsc )
			
		inputfiles = newfiles
	
	E2end(logger)
	
	return


def calcangles( f ):

	nf = EMUtil.get_image_count( f )
	angles = []
	angle = None

	for i in range( nf ):
		imghdr = EMData( f, i, True )
		try:
			angle = imghdr['spt_tiltangle']
			angles.append( angle ) 
		except:
			print """\n(e2spt_fillwedge.py)(calcangles) ERROR: image %d in stack %s lacking spt_tiltangle parameter in header.
				If tilt angle information is not in the header of the data, supply it via --tltfile.""" %( i, f )
			sys.exit()	

	angles.sort()
		
	return angles


def getangles( options ):
	print "\n(e2spt_fillwedge.py)(getangles)"
	angles = []
	
	print "Reading tlt file", options.tltfile
	f = open( options.tltfile, 'r' )
	lines = f.readlines()
	f.close()
	
	for line in lines:
		line = line.replace('\t','').replace('\n','')
		if line:
			angles.append( float(line) )
		print "Found angle", line
		
	if options.verbose > 9:
		print "\n(e2spt_ctf.py)(getangles) angles are", angles

	return angles


def writeanglestoheader( f, angs ):
	print "\n(e2spt_fillwedge.py)(writeanglestoheader)"
	n = EMUtil.get_image_count(f)
	print "Working on file", f
	print "With these many images and angles", n, len(angs)
	for i in range(n):
		print "Working on image", i
		imghdr = EMData( f, i, True)
		imghdr['spt_tiltangle']=angs[i]
		imghdr.write_image( f, i, EMUtil.ImageType.IMAGE_HDF, True )
		
	return 1
	
	
def writeanglestofile( options, angsf ):
	lines = []
	for a in angsf:
		line = str(angsf)+'\n'
		lines.append(line)
	finalangsfile = options.path + '/completeangles.tlt'
	gg=open(finalangsfile,'w')
	gg.writelines( lines )
	gg.close()
	return


def completeangles( options, angles ):
	
	#print "\n(e2spt_fillwedge.py)(completenalges) len(angles) is", len(angles)			
	anglesdifsum = sum( [ abs(angles[x] - angles[x+1]) for x in xrange(0,len(angles)-1) ] )			#C:Sum of differences between all consecutive tilts
	tiltstep = int( round( anglesdifsum / float(len(angles)) ) )									#C:Find the average tilt step
	
	if options.verbose > 5:
		print "\n(e2spt_fillwedge.py)(completenalges) The average tiltstep is", tiltstep
	lowerlimit = int( floor (min ( angles ) ) )
	upperlimit = int( ceil (max ( angles ) ) )
	
	lowerset = set([])
	if lowerlimit > -90:
		lowerset = [ float(v*-1) for v in xrange( lowerlimit*-1 + tiltstep, -90*-1, tiltstep ) ]
		lowerset.sort()
	
	upperset = set([])
	if upperlimit < 90:
		upperset = [ float(v) for v in xrange( upperlimit + tiltstep, 91, tiltstep ) ]
		upperset.sort()
	
	allangles = set().union( set(lowerset) ).union( set(angles) ).union( set(upperset) )
	allangles = list( allangles )
	allangles.sort()
	
	return ( lowerset, upperset, allangles, tiltstep )


def makevol( options, f , it, lowmostangle = -90, upmostangle = 90, writevols = 1 ):
		
	mode='gauss_2'
	if options.reconstructor:
		if len(options.reconstructor) > 1:
			if 'mode' in options.reconstructor[-1]:
				mode=options.reconstructor[-1]['mode']
		
	hdr = EMData( f, 0, True )
	apix = hdr['apix_x']
	
	nx = int( hdr['nx'] )
	ny = int( hdr['ny'] )
	xc = nx/2
	yc = ny/2

	originalsize = newsize = max( nx, ny )
	if options.pad3d:
		newsize = originalsize * options.pad3d
		newsize = int( round( newsize ) )
		if options.verbose > 5:
			print "\nPPPPPPPPPPPPPPP\n\n(e2spt_fillwedge.py)(makevol) Because there's options.pad=%f,then newsize=%d" % ( options.pad3d, newsize )
	
	if options.verbose > 9:
		print "\n(e2spt_fillwedge.py)(makevol) Mode for reconstructor is", mode
		print "Setting up reconstructor; options.reconstructor is", options.reconstructor
		print "For volume of size",newsize,newsize,newsize
		
	r = Reconstructors.get(options.reconstructor[0],{'size':(newsize,newsize,newsize),'sym':'c1','verbose':True,'mode':mode})
	r.setup()
	
	nf = EMUtil.get_image_count( f )
	print "\nNumber of images in file %s is %d" %( f, nf )
	
	axis = 'y'
	if options.tiltaxis:
		axis = options.tiltaxis
	
	angles = []
	
	print "\n++++++++++++++++++++++++++++++\n(makevol) file=%s, lowmostRAWangle=%f, upmostRAWangle=%f" % ( f, lowmostangle, upmostangle )
	print "However, adding projections, the total number of images will be", nf
	print "\n++++++++++++++++++++++++++++++\n"
	
	for i in range( nf ):
		angle = None
		#print "iter is", it
		#print "i is", i
		#print "f is ",f
		
		if options.verbose > 9:
			print  "\n(e2spt_fillwedge.py)(makevol) processing image %d in stack %s and ITERATION %d" % ( i, f, it )
		img = EMData( f, i )
		if options.verbose > 9:
			print "For image %d, mean=%f, min=%f, max=%f, std=%f" % ( i, img['mean'], img['minimum'],img['maximum'],img['sigma'] )
		
		#rad = img['nx']
		#img.process_inplace('mask.sharp',{'outer_radius':-1})
		
		if options.pad2d and float(options.pad2d) > 1.0:
		
			#R = Region( (2*xc - newsize)/2, (2*yc - newsize)/2, 0, newsize , newsize , 1)
			#img.clip_inplace( R )
			
			img = clip2D( img, newsize )
			#print "\nPadding prj to newsize", newsize
		
		try:
			angle = img['spt_tiltangle'] 
		except:
			print "\nERROR: 'spt_tiltangle' parameter not found in the header of image %d in file %s" % ( i, f )
			sys.exit()
		
		try:
			axis = img['spt_tiltaxis']
		except:
			if options.verbose > 9:
				print """\n(e2spt_fillwedge.py)(makevol) WARNING: 
				No spt_tiltaxis or sptsim_tiltaxis found in header. Default used.""", options.tiltaxis, axis
		
		if angle != None:
			angles.append( angle )
		
			t = Transform({'type':'eman','az':90,'alt':angle,'phi':-90})
		
			if axis == 'x':
				t = Transform({'type':'eman','alt':angle})
			
			#if options.preproc2d:
			#	filling=False
			#	if int(angle) < int(lowmostangle) or int(angle) > int(upmostangle):
			#		filling = True
			#		print "\nFilling is ON for angle!", angle
			#	img = preprocImg( img, options, filling )
			
			imgp = r.preprocess_slice( img, t)
			
			
			
			weight = 1.0
			#if float(angle) < float(lowmostangle) or float(angle) > float(upmostangle):
			#	weight = calcWeight( angle, it, options )
				#prj.mult( weight )
				
				#if math.fabs( angle ) == 90.0:
				#	complement = 1.0 - math.fabs( math.cos( math.radians(89.99) ) )
				#	weight = math.fabs( (it+1) * math.cos( math.radians(89.99) ) / float(options.iter) ) + ( float(it)/float(options.iter)) * complement
				#else:
				#	complement = 1.0 - math.fabs( math.cos( math.radians(angle) ) )
				#	weight = math.fabs( (it+1) * math.cos( math.radians(angle) ) / float(options.iter) ) + (float(it)/float(options.iter)) * complement 
				
				#weight = math.fabs( weight )
			if options.verbose > 9:
				print "\n(makevol) ITER=%d, tiltangle=%f, weight=%f" % ( it, angle, weight )
				print "Inserted IMAGE with this tranform", t
			r.insert_slice( imgp , t , weight )
	
	rec = r.finish(True)

	rec['apix_x']=apix
	rec['apix_y']=apix
	rec['apix_z']=apix
	rec['origin_x']=0
	rec['origin_y']=0
	rec['origin_z']=0
	
	angles.sort()
	if options.verbose > 9:
		print "\n(e2spt_fillwedge.py)(makevol) Sorted angles to write to header are", angles
	
	rec['spt_tiltangles'] = angles

	recxc = rec['nx']/2
	recyc = rec['ny']/2
	reczc = rec['nz']/2

	#R2 =  Region( (2*recxc - originalsize)/2, (2*recyc - originalsize)/2, (2*reczc - originalsize)/2, originalsize , originalsize , originalsize)
	#rec.clip_inplace( R2 )
	
	outx=outy=outz=originalsize
	if options.outxsize:
		outx = options.outxsize
	if options.outysize:
		outy = options.outysize
	if options.outzsize:
		outz = options.outzsize
	
	rec = clip3D( rec, outx, outy, outz )	
	
	rec.process_inplace( 'normalize' )
	
	if options.verbose > 9:
		print "\n(e2spt_fillwedge.py)(makevol) Reconstructed volume for file", f
	
	volfile = ''
	if options.savevols and writevols:
		volfile = f.replace('.hdf','_3D.hdf')
		if options.path not in volfile:
			volfile = options.path + '/' + volfile

		rec.write_image( volfile, 0 )
		
		#if options.iter and int( options.iter ) > 1 and it == options.iter -1:
		#	if options.finalboxsize:
		#		rec = clip3D( rec, options.finalboxsize )
		#		rec.write_image( volfile, 0 )
		
	return ( rec, volfile )


def clip3D( vol, sizex, sizey=0, sizez=0 ):
	
	if not sizey:
		sizey=sizex
	
	if not sizez:
		sizez=sizex
	
	volxc = vol['nx']/2
	volyc = vol['ny']/2
	volzc = vol['nz']/2
	
	Rvol =  Region( (2*volxc - sizex)/2, (2*volyc - sizey)/2, (2*volzc - sizez)/2, sizex , sizey , sizez)
	vol.clip_inplace( Rvol )
	#vol.process_inplace('mask.sharp',{'outer_radius':-1})
	
	return vol


def clip2D( img, size ):
	
	imgxc = img['nx']/2
	imgyc = img['ny']/2
	#imgzc = img['nz']/2
	
	Rimg =  Region( (2*imgxc - size)/2, (2*imgyc - size)/2, 0, size , size , 1)
	img.clip_inplace( Rimg )
	#img.process_inplace('mask.sharp',{'outer_radius':-1})
	
	return img


def reprojectvolume():
	pass
	return
	

def genprojection( vol, angle, tiltaxis ):

	t = Transform({'type':'eman','az':90,'alt':angle,'phi':-90})
		
	if tiltaxis == 'x':
		t = Transform({'type':'eman','alt':angle})
	
	prj = vol.project("standard",t)
	#prj.process_inplace('normalize.edgemean')
	prj['spt_tiltangle'] = angle

	return prj



def preprocImg( img, options, filling=False ):
	
	img.process_inplace('normalize.edgemean')
	
	if filling and options.preprocessfilling and options.preprocessfilling != 'None' and options.preprocessfilling != 'none':
		preprocessfilling=''
		try:
			preprocessfilling=parsemodopt(options.preprocessfilling)
		except:
			pass	
		img.process_inplace( preprocessfilling[0], preprocessfilling[1] )
	elif options.preprocess and options.preprocess != 'None' and options.preprocess != 'none': 
		preprocess=''
		try:
			preprocess=parsemodopt(options.preprocess)
		except:
			pass
		img.process_inplace( preprocess[0], preprocess[1] )
		
	
	if filling and options.maskfilling and options.maskfilling != 'None' and options.maskfilling != 'none':
		maskfilling=''
		try:
			maskfilling=parsemodopt(options.maskfilling)
		except:
			pass
		img.process_inplace( maskfilling[0], maskfilling[1] )
	elif options.mask and options.mask != 'None' and options.mask != 'none':
		mask=''
		try:
			mask=parsemodopt(options.mask)
		except:
			pass
		img.process_inplace( mask[0], mask[1] )
	
	
	if filling and options.thresholdfilling and options.thresholdfilling != 'None' and options.thresholdfilling != 'none':
		thresholdfilling=''
		try:
			thresholdfilling=parsemodopt(options.thresholdfilling)
		except:
			print "Failed to parse threshold"	
		print "Parsed threshold is", thresholdfilling
		img.process_inplace( thresholdfilling[0], thresholdfilling[1] )	
	elif options.threshold and options.threshold != 'None' and options.threshold != 'none': 
		threshold=''
		try:
			threshold=parsemodopt(options.threshold)
		except:
			print "Failed to parse threshold"
		print "Parsed threshold is", threshold
		img.process_inplace( threshold[0], threshold[1] )

	
	if filling and options.highpassfilling and options.highpassfilling != 'None' and options.highpassfilling != 'none':
		highpassfilling=''
		try:
			highpassfilling=parsemodopt(options.highpassfilling)
		except:
			pass
		img.process_inplace( highpassfilling[0], highpassfilling[1] )
	elif options.highpass and options.highpass != 'None' and options.highpass != 'none': 
		highpass=''
		try:
			highpass=parsemodopt(options.highpass)
		except:
			pass
		img.process_inplace( highpass[0], highpass[1] )
	
	
	if filling and options.lowpassfilling  and options.lowpassfilling != 'None' and options.lowpassfilling != 'none':
		lowpassfilling=''
		try:
			lowpassfilling=parsemodopt(options.lowpassfilling)
		except:
			pass
		img.process_inplace( lowpassfilling[0], lowpassfilling[1] )
	elif options.lowpass and options.lowpass != 'None' and options.lowpass != 'none': 
		lowpass=''
		try:
			lowpass=parsemodopt(options.lowpass)
		except:
			pass
		img.process_inplace( lowpass[0], lowpass[1] )
	
	return img


def calcWeight( angle, it, options ):

	weight = 1.0
	#if float(angle) < float(lowmostangle) or float(angle) > float(upmostangle):
	if math.fabs( angle ) == 90.0:
		complement = 1.0 - math.fabs( math.cos( math.radians(89.99) ) )
		weight = math.fabs( (it+1) * math.cos( math.radians(89.99) ) / float(options.iter) ) + ( float(it)/float(options.iter)) * complement
	else:
		complement = 1.0 - math.fabs( math.cos( math.radians(angle) ) )
		weight = math.fabs( (it+1) * math.cos( math.radians(angle) ) / float(options.iter) ) + (float(it)/float(options.iter)) * complement 

	print "Something was weighed!!!!"
	
	return weight
	

def wedgefiller( options, originalseries, vol, originalangles, lowerangles, upperangles, allangles, tiltstep, it):
	#options, originalseries, vol, originalangles, lowerangles, upperangles, allangles, tiltstep, iter
	
	nimgsOriginal = EMUtil.get_image_count( originalseries )
	middleIndx = nimgsOriginal/2
	
	middleSlice = EMData( originalseries, middleIndx )
	
	if options.preproc2d:
		middleSlice = preprocImg( middleSlice, options, False )
		
	volnx = middleSlice['nx']
	volny = middleSlice['ny']
	
	try:
		volnz = vol['nz']
	except:
		volnz = volnx
		
	originalvolsize = max( volnx, volny, volnz)
	expandedvolsize = originalvolsize+2
		
	currentslicefile = options.path + '/currentslice.hdf'
	tmpfile = options.path + '/tmp.hdf'
	
	nLangles = len( lowerangles )
	nUangles = len (upperangles )
	mostangles = max( nLangles, nUangles )
	
	tiltaxis = 'y'
	if options.tiltaxis:
		tiltaxis = options.tiltaxis
	
	k=0
	writevols = 0
	loweranglescopy = list( lowerangles )
	loweranglescopy.sort()
	loweranglescopy.reverse()
	
	if options.preproc2d and '_fwpreproc' not in originalseries:
		
		originalseriespreproc = originalseries.replace('.hdf','_fwpreproc.hdf')
		
		if options.path not in originalseriespreproc:
			originalseriespreproc = options.path + '/' + originalseriespreproc
		
		print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!originalseriespreproc is", originalseriespreproc
		
		try:
			os.system('rm ' + options.path + '/' + originalseriespreproc)
		except:
			pass
			
		preproccmd = 'e2proc2d.py ' + originalseries + ' ' + originalseriespreproc 
		
		if options.mask:
			preproccmd+= " --process=" + options.mask
		if options.lowpass:
			preproccmd+= " --process=" + options.lowpass
		if options.highpass:
			preproccmd+= " --process=" + options.highpass
		if options.preprocess:
			preproccmd+= " --process=" + options.preprocess
		if options.threshold:
			preproccmd+= " --process=" + options.threshold
		
		print "\npreproccmd is", preproccmd
		p=subprocess.Popen( preproccmd , shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)		
		while True:
			out = p.stdout.read(1)
			if out == '' and p.poll() != None:
				break
			if out != '':
				sys.stdout.write(out)
				sys.stdout.flush()
		p.stdout.close()
		
		originalseries = originalseriespreproc
		
	
	for j in range(mostangles):
		
		vol = clip3D( vol, expandedvolsize )
		
		vol = preprocImg( vol, options, True )
	
		lowerangle = upperangle = None
		if j < nLangles:
			lowerangle = loweranglescopy[j]
			if options.verbose > 9:
				print "\nIn iteration %d lowerangle is %f" % ( it, lowerangle )
		if j < nUangles:
			upperangle = upperangles[j]
			if options.verbose > 9:
				print "\nIn iteration %d upperangle is %f" % ( it , upperangle )
		
		if j == mostangles - 1:
			if options.savevols:
				writevols = 1
				
		if lowerangle:
			#vol = clip3D( vol, expandedvolsize )
			#vol.process_inplace('mask.sharp',{'outer_radius':-1})
		
			prj = genprojection( vol, lowerangle, tiltaxis )
			prj = clip2D( prj, originalvolsize )
			#prj.process_inplace('normalize.edgemean')
		
			prj.process_inplace('normalize.edgemean')
			
			if options.preproc2d:
				prj = preprocImg( prj, options, True )
			
			print "\nMiddleslicemax is", middleSlice['maximum']
			
			print "for prj max is", prj['maximum']
			
			highth=prj["mean"]+prj["sigma"]*3.0
			prj.process_inplace('filter.matchto',{'to':middleSlice})
			
			print "After matchto it is", prj['maximum']
			
			#if options.normalizetomiddleslice:
			#	prj.process_inplace("normalize.toimage",{"to":ptcl,"ignore_lowsig":0.75,"high_threshold":highth})
			
			#prj.process_inplace("normalize.toimage",{"to":middleSlice,"ignore_lowsig":0.75,"high_threshold":highth})
			
			
			#prj.process_inplace("normalize.toimage",{"to":middleSlice})
			print "After normalize to middle slice it is", prj['maximum']
			
			if options.weightbyangle:
				weight = calcWeight( lowerangle, it, options )
				prj.mult( weight )
			
			#prj.process_inplace('filter.matchto',{'to':middleSlice})			
			prj.write_image( currentslicefile, 0 )
		
			if k == 0:
				n = EMUtil.get_image_count( originalseries )
				for ii in range( n ):
					EMData( originalseries, ii ).write_image( currentslicefile, -1 )
				
				cmd = 'mv ' + currentslicefile + ' ' + tmpfile + ' && rm ' + currentslicefile
				os.popen( cmd )
			else:
				n = EMUtil.get_image_count( tmpfile )
				for ii in range( n ):
					EMData( tmpfile, ii ).write_image( currentslicefile, -1 )
				
				cmd = 'rm ' + tmpfile + ' && mv ' + currentslicefile + ' ' + tmpfile
				os.popen( cmd )
		
			retm = makevol( options, tmpfile, it, originalangles[0], originalangles[-1], writevols )
			vol = retm[0]
			volfile = retm[1]
			
			k+=1
		
		if upperangle:
			#vol = clip3D( vol, expandedvolsize )
			#vol.process_inplace('mask.sharp',{'outer_radius':-1})
		
			prj = genprojection( vol, upperangle, tiltaxis )
			prj = clip2D( prj, originalvolsize )	
			#prj.process_inplace('normalize.edgemean')
			
			prj.process_inplace('normalize.edgemean')
			
			if options.preproc2d:
				prj = preprocImg( prj, options, True )
			
			highth=prj["mean"]+prj["sigma"]*3.0
			prj.process_inplace('filter.matchto',{'to':middleSlice})
			#projf.process_inplace("normalize.toimage",{"to":ptcl,"ignore_lowsig":0.75,"high_threshold":highth})
			
			if options.normalizetomiddleslice:
				prj.process_inplace("normalize.toimage",{"to":middleSlice,"ignore_lowsig":0.75,"high_threshold":highth})
			
			if options.weightbyangle:
				weight = calcWeight( upperangle, it, options )
				prj.mult( weight )
			
			#prj.process_inplace('filter.matchto',{'to':middleSlice})
			prj.write_image( currentslicefile, 0 )
		
			if k == 0:
				cmd = 'cp ' + originalseries + ' ' + tmpfile
				os.popen( cmd )
				
			EMData( currentslicefile, 0 ).write_image( tmpfile, -1 )
			cmd = 'rm ' + currentslicefile
			os.popen( cmd )
			
			#else:
			#	cmd = 'e2proc2d.py ' + currentslicefile + ' ' + tmpfile
			#	cmd += ' && rm ' + currentslicefile
			#	os.popen( cmd )
		
			retm = makevol( options, tmpfile, it, originalangles[0], originalangles[-1], writevols )
			vol = retm[0]
			volfile = retm[1]
			k+=1
			
	#completedseries = originalseries.replace('.hdf','_FULL.hdf')
	#if options.iter > 1:
	#reprojectedseries = os.path.basename( f )
	
	completedseries = options.path + '/' + os.path.basename( originalseries )
	if "_FULL" not in completedseries:
		completedseries = completedseries.replace( '.hdf', '_FULL.hdf' )
		
	cmd = 'mv ' + tmpfile + ' ' + completedseries
	finalvolfile=volfile
	if options.savevols and volfile:
		newvolfile = completedseries.replace( 'FULL.hdf', 'FULL3D.hdf' )
		cmd += ' && mv ' + volfile + ' ' + newvolfile
		finalvolfile=newvolfile
		
	os.system( cmd )
	
	return ( completedseries, vol, finalvolfile )
	
	
if __name__ == '__main__':
	main()

