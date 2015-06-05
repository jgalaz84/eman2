/**
 * $Id: xydata.cpp,v 1.24 2014/04/29 19:24:25 stevel Exp $
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

#include "xydata.h"
#include <algorithm>
#ifndef WIN32
#include <sys/param.h>
#else
#include <windows.h>
#define MAXPATHLEN (MAX_PATH*4)
#endif
#include <cfloat>
#include <cmath>
#include <cstdio>
#include "log.h"
#include "exception.h"

using namespace EMAN;

XYData::XYData()
	: ymin(FLT_MAX), ymax(-FLT_MAX), mean_x_spacing(1.0)
{
}

void XYData::update()
{
	if (data.size()==0) return;
	
	std::sort(data.begin(), data.end());

	ymin = FLT_MAX;
	ymax = -FLT_MAX;

	typedef vector < Pair >::const_iterator Ptype;
	for (Ptype p = data.begin(); p != data.end(); p++) {
		if (p->y > ymax) {
			ymax = p->y;
		}
		if (p->y < ymin) {
			ymin = p->y;
		}
	}

	size_t n = data.size();
	mean_x_spacing = (data[n - 1].x - data[0].x) / (float) n;
}


int XYData::read_file(const string & filename)
{
	FILE *in = fopen(filename.c_str(), "rb");
	if (!in) {
		throw FileAccessException(filename.c_str());
//		LOGERR("cannot open xydata file '%s'", filename.c_str());
//		return 1;
	}

	char buf[MAXPATHLEN];
	char tmp_str[MAXPATHLEN];

	while (fgets(buf, MAXPATHLEN, in)) {
		if (buf[0] != '#') {
			float x = 0;
			float y = 0;

			if (sscanf(buf, " %f%[^.0-9-]%f", &x, tmp_str, &y) != 3) {
				break;
			}
			data.push_back(Pair(x, y));
		}
	}

	fclose(in);
	in = 0;

	update();

	return 0;
}

int XYData::write_file(const string & filename) const
{
	FILE *out = fopen(filename.c_str(), "wb");
	if (!out) {
		LOGERR("cannot open xydata file '%s' to write", filename.c_str());
		return 1;
	}

	typedef vector < Pair >::const_iterator Ptype;
	for (Ptype p = data.begin(); p != data.end(); p++) {
		fprintf(out, "%1.6g\t%1.6g\n", p->x, p->y);
	}

	fclose(out);
	out = 0;

	return 0;
}


float XYData::calc_correlation(XYData * xy, float minx, float maxx) const
{
	size_t n = data.size();
	float x0 = data[0].x;
	float xn = data[n - 1].x;

	if (maxx <= minx || minx >= xn || maxx <= x0) {
		LOGERR("incorrect minx, maxx=%f,%f for this XYData range [%f,%f]",
			   minx, maxx, x0, xn);
		return 0;
	}

	float scc = 0;
	float norm1 = 0;
	float norm2 = 0;

	xy->update();
	for (size_t i = 0; i < n; i++) {
		float x = data[i].x;
		if (x >= minx && x <= maxx && xy->is_validx(x)) {
			float selfy = data[i].y;
			float xyy = xy->get_yatx(x);

			scc += selfy * xyy;
			norm1 += selfy * selfy;
			norm2 += xyy * xyy;
		}
	}

	float result = scc / sqrt(norm1 * norm2);
	return result;
}


float XYData::get_yatx(float x,bool outzero)
{
	if (data.size()==0 || mean_x_spacing==0) return 0.0;

	int nx = (int) data.size();
	// Do the range checking up front before we get into trouble
	if (x<data[0].x) return outzero?0.0f:data[0].y;
	if (x>data[nx-1].x) return outzero?0.0f:data[nx-1].y;
	
	int s = (int) floor((x - data[0].x) / mean_x_spacing);
	if (s>nx-1) s=nx-1;

	// These deal with possibly nonuniform x values. A btree would be better, but this is simple
	while (s>0 && data[s].x > x) s--;
	while (s<(nx-1) && data[s + 1].x < x ) s++;
	if (s>nx-2) return outzero?0.0f:data[nx-1].y;
	
	float f = (x - data[s].x) / (data[s + 1].x - data[s].x);
	float y = data[s].y * (1 - f) + data[s + 1].y * f;
	return y;
}

float XYData::get_yatx_smooth(float x,int smoothing)
{
	if (data.size()==0) return 0.0;
	if (data.size()==1) return data[0].y;
	if (smoothing!=1) throw InvalidParameterException("Only smoothing==1 (linear) currently supported");
	
	int nx = (int) data.size();
	
	int s = nx/2; 
	if (mean_x_spacing>0) s=(int) floor((x - data[0].x) / mean_x_spacing);
	if (s>nx-2) s=nx-2;
	else if (s<0) s=0;
	else {
		// These deal with possibly nonuniform x values. A btree would be better, but this is simple, and there usually won't be THAT many points
		while (s>0 && data[s].x > x) s--;
		while (s<(nx-2) && data[s + 1].x < x ) s++;
	}
	
	float f = 0,y=0;
	if (data[s + 1].x != data[s].x) {
		f= (x - data[s].x) / (data[s + 1].x - data[s].x);
		y = data[s].y * (1 - f) + data[s + 1].y * f;
	}
	else {
		int s2=s;
		while (data[s2].x==data[s].x) {
			if (s2<nx-1) s2++;
			if (s>0) s--;
			if (s==0 &&s2==nx-1) return data[nx/2].y;
		}
		f= (x - data[s].x) / (data[s2].x - data[s].x);
		y = data[s].y * (1 - f) + data[s2].y * f;
		
	}
//	printf("%d %1.2f x %1.2f %1.2f %1.2f y %1.2f %1.2f\n",s,f,x,data[s].x,data[s+1].x,data[s].y,data[s+1].y);
	return y;
}

// FIXME: clearly a slow way to do this
void XYData::insort(float x, float y) {
	data.push_back(Pair(x,y));
	update();
	
	
	// 	int nx = (int) data.size();
// 	if (nx==0) { data.push_back(Pair(x,y)); return; }
// 	
// 	int s = (int) floor((x - data[0].x) / mean_x_spacing);
// 	if (s>nx) s=nx;
// 	else if (s<0) s=0;
// 	else {
// 	// These deal with possibly nonuniform x values. A btree would be better, but this is simple, and there usually won't be THAT many points
// 		while (s>0 && data[s-1].x > x) s--;
// 		while (s<nx && data[s].x <= x ) s++;
// 	}
// 	data.insert(data.begin()+s,Pair(x,y));
}	

// data must be sorted before this will work
void XYData::dedupx() {
	float acnt=1.0;
	for (std::vector<Pair>::iterator it = data.begin() ; it+1 < data.end(); ++it) {
		while (it+1<data.end() && it->x==(it+1)->x) {
//			printf("%d\t%d\t%1.0f\t%f\t%f\n",it-data.begin(),data.end()-it,acnt,it->x,(it+1)->x);
			it->y+=(it+1)->y;
			acnt+=1.0;
			data.erase(it+1);
		}
		if (acnt>1.0) {
			it->y/=acnt;
			acnt=1.0;
		}
	}
}

void XYData::set_xy_list(const vector<float>& xlist, const vector<float>& ylist)
{
	if(xlist.size() != ylist.size()) {
		throw InvalidParameterException("xlist and ylist size does not match!");
	}

	for(unsigned int i=0; i<xlist.size(); ++i) {
		data.push_back(Pair(xlist[i], ylist[i]));
	}
}

void XYData::set_size(size_t n)
{
	data.resize(n, Pair(0.0f, 0.0f));
}

vector<float>  XYData::get_state() const {
	vector<float> list;
	vector<Pair>::const_iterator cit;
	for(cit=data.begin(); cit!=data.end(); ++cit) {
		list.push_back( (*cit).x);
		list.push_back( (*cit).y);
	}

	return list;
	
}

void  XYData::set_state(vector<float> list) {
	if(list.size()%2==1) {
		throw InvalidParameterException("Invalid pickled data");
	}

	data.clear();
	for(unsigned int i=0; i<list.size(); i+=2) {
		data.push_back(Pair(list[i], list[i+1]));
	}

	update();
}

vector<float> XYData::get_xlist() const
{
	vector<float> xlist;
	vector<Pair>::const_iterator cit;
	for(cit=data.begin(); cit!=data.end(); ++cit) {
		xlist.push_back( (*cit).x);
	}

	return xlist;
}

vector<float> XYData::get_ylist() const
{
	vector<float> ylist;
	vector<Pair>::const_iterator cit;
	for(cit=data.begin(); cit!=data.end(); ++cit) {
		ylist.push_back( (*cit).y);
	}

	return ylist;
}
