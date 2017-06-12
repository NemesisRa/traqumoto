/* Ce fichier se lance via l'executable. Le Main initialise Lua, charge les librairies Lua, y ajoute la fonction getFile et execute le fichier traqumoto.lua */

#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

extern "C" {	/* charge les librairies Lua */
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lua.h"
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lualib.h"
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lauxlib.h"
}

/* interpreteur */
lua_State* L;

/* La fonction getFile permet de recupérer la vidéo à traiter */
static int getFile(lua_State *L)
{

	FILE *in;
	if (!(in = popen("zenity  --title=\"Select an image\" --file-selection","r")))
	{
    		return 1;
	}

	char buff[512];
	string selectFile = "";
	while (fgets(buff, sizeof(buff), in) != NULL)
	{
		selectFile += buff;
	}
	pclose(in);

	/* push the return */
	lua_pushstring(L, selectFile.c_str());

	/* return the number of results */
	return 1;
}

int main ( int argc, char *argv[] )
{
	/* initialise Lua */
	L = lua_open();

	/* charge les librairies de base de Lua */
	luaL_openlibs(L);

	/* enregistre la fonction getFile dans la librairie Lua */
	lua_register(L, "getFile", getFile);

	/* execute le script */
	luaL_dofile(L, "src/traqumoto.lua");

	/* ferme Lua */
	lua_close(L);

	return 0;
}
