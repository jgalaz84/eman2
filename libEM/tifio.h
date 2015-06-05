/**
 * $Id: tifio.h,v 1.16 2014/04/29 21:35:07 vern Exp $
 */

/*
 * Author: Steven Ludtke, 04/10/2003 (sludtke@bcm.edu)
 * Copyright (c) 2000-2006 Baylor College of Medicine
 * 
 * This software is issued under a joint BSD/GNU license. You may use the
 * source code in this file under either license. However, note that the
 * complete EMAN2 and SPARX software packages have some GPL dependencies,
 * so you are responsible for compliance with the licenses of these packages
 * if you opt to use BSD licensing. The warranty disclaimer below holds
 * in either instance.
 * 
 * This complete copyright notice must be included in any revised version of the
 * source code. Additional authorship citations may be added, but existing
 * author citations must be preserved.
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * */

#ifndef eman__tiffio_h__
#define eman__tiffio_h__ 1

#ifdef EM_TIFF

#include "imageio.h"

typedef struct tiff TIFF;

namespace EMAN
{
	/** TiffIO reads/writes a TIFF image. 8-bit and 16-bit TIFF images
	 * are supported so far.
	 *
	 * A TIFF file contains 1 2D image.
	*/
	class TiffIO:public ImageIO
	{
	  public:
		explicit TiffIO(string filename, IOMode rw_mode = READ_ONLY);
		~TiffIO();

		DEFINE_IMAGEIO_FUNC;
		static bool is_valid(const void *first_block);
		int get_nimg();

	  private:
		enum
		{
			TIFF_LITTLE_ENDIAN = 0x49,
			TIFF_BIG_ENDIAN = 0x4d
		};

		string filename;
		IOMode rw_mode;
		TIFF *tiff_file;
		int nimg;
		unsigned short bitspersample;
		unsigned short photometric;
		bool is_big_endian;
		bool initialized;
		
		unsigned int nx;
		unsigned int ny;
		unsigned int nz;
		
		float rendermin;
		float rendermax;
	};
}

#endif	//EM_TIFF

#endif	//eman__tiffio_h__
