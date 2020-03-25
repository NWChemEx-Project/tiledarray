function(clone_imported_target dst src)
  add_library(${dst} INTERFACE IMPORTED)
  foreach(name
      IMPORTED_COMMON_LANGUAGE_RUNTIME
      IMPORTED_CONFIGURATIONS
      IMPORTED_IMPLIB
      IMPORTED_LIBNAME
      IMPORTED_LINK_DEPENDENT_LIBRARIES
      IMPORTED_LINK_INTERFACE_LIBRARIES
      IMPORTED_LINK_INTERFACE_MULTIPLICITY
      IMPORTED_NO_SONAME
      IMPORTED_OBJECTS
      IMPORTED_SONAME
      IMPORT_PREFIX
      IMPORT_SUFFIX

      INTERFACE_AUTOUIC_OPTIONS
      INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_FEATURES
      INTERFACE_COMPILE_OPTIONS
      INTERFACE_INCLUDE_DIRECTORIES
      INTERFACE_LINK_DEPENDS
      INTERFACE_LINK_DIRECTORIES
      INTERFACE_LINK_LIBRARIES
      INTERFACE_LINK_OPTIONS
      INTERFACE_PRECOMPILE_HEADERS
      INTERFACE_POSITION_INDEPENDENT_CODE
      INTERFACE_SOURCES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES

      LINK_INTERFACE_LIBRARIES
      LINK_INTERFACE_MULTIPLICITY

      MAP_IMPORTED_CONFIG

      NO_SYSTEM_FROM_IMPORTED
      )
    foreach(config "" "_DEBUG" "_RELWITHDEBINFO" "_RELEASE" "_MINSIZEREL")
      set(prop ${name}${config})
      get_property(value_set TARGET ${src} PROPERTY ${prop} SET)
      if (value_set)
        get_property(value TARGET ${src} PROPERTY ${prop})
        set_property(TARGET ${dst} PROPERTY ${prop} ${value})
      endif(value_set)
    endforeach()
  endforeach()
endfunction()

function(clone_target dst src)
  add_library(${dst} INTERFACE)
  foreach(name
      INTERFACE_AUTOUIC_OPTIONS
      INTERFACE_COMPILE_DEFINITIONS
      INTERFACE_COMPILE_FEATURES
      INTERFACE_COMPILE_OPTIONS
      INTERFACE_INCLUDE_DIRECTORIES
      INTERFACE_LINK_DEPENDS
      INTERFACE_LINK_DIRECTORIES
      INTERFACE_LINK_LIBRARIES
      INTERFACE_LINK_OPTIONS
      INTERFACE_PRECOMPILE_HEADERS
      INTERFACE_POSITION_INDEPENDENT_CODE
      INTERFACE_SOURCES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES

      LINK_INTERFACE_LIBRARIES
      LINK_INTERFACE_MULTIPLICITY
      )
    foreach(config "" "_DEBUG" "_RELWITHDEBINFO" "_RELEASE" "_MINSIZEREL")
      set(prop ${name}${config})
      get_property(value_set TARGET ${src} PROPERTY ${prop} SET)
      if (value_set)
        get_property(value TARGET ${src} PROPERTY ${prop})
        set_property(TARGET ${dst} PROPERTY ${prop} ${value})
      endif(value_set)
    endforeach()
  endforeach()
endfunction()
