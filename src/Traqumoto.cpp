#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

extern "C" {
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lua.h"
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lualib.h"
	#include "../Applications/torch/exe/luajit-rocks/luajit-2.1/src/lauxlib.h"
}

/* the Lua interpreter */
lua_State* L;

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
	/* initialize Lua */
	L = lua_open();

	/* load Lua base libraries */
	luaL_openlibs(L);

	/* register our function */
	lua_register(L, "getFile", getFile);

	/* run the script */
	luaL_dofile(L, "src/Traqumoto.lua");

	/* cleanup Lua */
	lua_close(L);

	return 0;
}
