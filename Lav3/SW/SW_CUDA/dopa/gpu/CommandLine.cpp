/**
A couple of functions to automatically show/read command line arguments

Copyright 2010 Marijn Kentie
 
This file is part of GASW.

GASW is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GASW is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GASW.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "CommandLine.h"
#include "main.h"

struct cmdArg
{
	void* ptr;
	int isString;
	char *flag;
	char *description;
};


struct cmdArg args[] = {{&options.matrix,true,"-s","Substitution matrix"},
{&options.gapWeight,false,"-f","Gap penalty"},
{&options.extendWeight,false,"-g","Gap extend penalty"},
{&options.listSize,false,"-b","Number of scores to show"},
{&options.topSequenceFile,true,"-o","Output database file for top scoring sequences"},
{&options.dna,false,"-d","DNA mode (-d 1 for DNA mode)"}
};

const unsigned int NUM_ARGS = sizeof(args)/sizeof(cmdArg);

/**
Read the program's command line into the supplied variables
*/
bool parseCommandLine(int argc, char* argv[])
{
	for(int i=1;i<argc;i++)
	{
		int argMatch=0;
		for(int j=0;j<NUM_ARGS;j++) //Check command line switches
		{
			if(strcmp(argv[i],args[j].flag)==0)
			{
				if(argc>i+1) //The switch has a parameter
				{
					//Read parameter into recipient variable
					if(args[j].isString)
						*((char**)args[j].ptr) = argv[i+1];
					else
						*((int*)args[j].ptr) = atoi(argv[i+1]);
					argMatch = true;
					break;
				}
				else
				{
					printf("Error in argument <%s>.\n",args[j].description);
					return false;
				}
			}
		}

		if(argMatch)
		{
			i++; //Skip parameter we just read
			continue;
		}

		//The current argument wasn't a switch; it's the in or output file
		if (!options.sequenceFile)
			options.sequenceFile = argv[i];
		else
			options.dbFile = argv[i];
	}
	
	return true;
}

/**
Show the supported arguments and their descriptions.
*/
void printArgs()
{
	for(int i=0;i<NUM_ARGS;i++)
	{
		char* argType = args[i].isString ? (char*)"<string>" : (char*)"<num>";
		printf("%s %s \t\t %s\n",args[i].flag,argType,args[i].description);
	}
}

/**
Show an overview of the current settings read from the command line.
*/
void printArgValues()
{

	for(int i=0;i<NUM_ARGS;i++)
	{
		printf("%s: ",args[i].description);
		if(args[i].isString)
			printf("%s\n",*((char**) args[i].ptr));
		else
			printf("%d\n",*((int*) args[i].ptr));
	}
}
