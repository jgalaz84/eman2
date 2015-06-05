/**
 * $Id: portable_fileio.h,v 1.14 2008/07/25 22:17:27 gtang Exp $
 */

/*
 * Author: Liwei Peng, 08/27/2003 (sludtke@bcm.edu)
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

#ifndef __portable_fileio_h__
#define __portable_fileio_h__

#include <cstdio>
#include <sys/types.h>


inline int portable_fseek(FILE * fp, off_t offset, int whence)
{
#if defined(HAVE_FSEEKO)
	return fseeko(fp, offset, whence);
#elif defined(HAVE_FSEEK64)
	return fseek64(fp, offset, whence);
#elif defined(__BEOS__)
	return _fseek(fp, offset, whence);
#else
	return fseek(fp, offset, whence);
#endif
}

inline off_t portable_ftell(FILE * fp)
{
#if defined(HAVE_FTELLO)
	return ftello(fp);
#elif defined(HAVE_FTELL64)
	return ftell64(fp);
#else
	return ftell(fp);
#endif
}



#endif
