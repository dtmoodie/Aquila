import argparse
import os

sections =[
    'project_id:',
    'include_dirs:',
    'lib_dirs_debug:',
    'lib_dirs_release:',
    'compile_options:',
    'compile_definitions:',
    'module_dependencies:',
    'compiler_location:',
    'link_libs:',
    'link_libs_debug:',
    'link_libs_release:'
]

parser = argparse.ArgumentParser()

parser.add_argument("--in_path", type=str)
parser.add_argument("--out_path", type=str)
parser.add_argument("--plugin_name", type=str)
parser.add_argument("--link_path", type=str)
args = parser.parse_args()

project_id = -1

link_file_lines = [x.rstrip() for x in open(args.link_path, 'rt').readlines()]

def pruneLibName(name):
    idx = name.find('.so')
    if idx != -1:
        name = name[:idx]
    if 'lib' in name[0:3]:
        name = name[3:]
    return name

def splitLib(name):
    filename = os.path.basename(name)
    dirname = os.path.dirname(name)
    filename = pruneLibName(filename)
    return dirname, filename


def removeDuplicates(x):
    out = []
    for i in x:
        if i not in out:
            out.append(i)
    return out


def parseSection(start_index, lines):
    j = start_index + 1
    for j in range(j, len(lines)):
        if lines[j].strip() in sections:
            break
    include_dirs = lines[start_index+1:j]
    out = []
    for x in include_dirs:
        x = [y.strip() for y in x.split(';')]
        for y in x:
            if(len(y)):
                out.append(y)
    return removeDuplicates(out)


def parsePathSection(start_index, lines):
    for j in range(start_index+1, len(lines)):
        if lines[j].strip() in sections:
            break
    include_dirs = lines[start_index+1:j]
    out = []
    for x in include_dirs:
        x = x.strip()
        if(os.path.exists(x)):
            out.append(x)
    return removeDuplicates(out)


def writeList(out, paths, prefix='', postfix=''):
    out.write('{\n')
    out.write('    static const char* paths[] = {')
    if(len(paths) == 0):
        out.write('\n        nullptr\n')
    else:
        for i, inc in enumerate(paths):
            if(i != 0):
                out.write(',')
            out.write('\n        "{}{}{}"'.format(prefix, inc, postfix))
        out.write(',\n        nullptr')
    out.write('\n    };\n')
    out.write('    return paths;\n')


with open(args.in_path, 'rt') as f:
    with open(args.out_path, 'wt') as out:
        with open(args.link_path, 'wt') as link_out:
            j = link_file_lines.index('ALL_LIBS')
            link_out.write('\n'.join(link_file_lines[:j]))
            link_file_lines = link_file_lines[j+1:]
            lines = f.readlines()
            for i in range(len(lines)):
                if('project_id' in lines[i]):
                    project_id = lines[i+1]
                if('include_dirs' in lines[i]):
                    include_dirs = parsePathSection(i, lines)
                if('lib_dirs_debug' in lines[i]):
                    debug_dirs = parsePathSection(i, lines)
                if('lib_dirs_release' in lines[i]):
                    release_dirs = parsePathSection(i, lines)
                if('compile_options' in lines[i]):
                    compile_opts = parseSection(i, lines)
                    compile_opts = [x if 'c++11' not in x else x.replace('c++11','gnu++14') for x in compile_opts]
                if('compile_definitions' in lines[i]):
                    compile_defs = parseSection(i, lines)
                    compile_defs = [x for x in compile_defs if 'NOTFOUND' not in x]
                if('module_dependencies' in lines[i]):
                    deps = parseSection(i, lines)
                if('compiler_location' in lines[i]):
                    cloc = parseSection(i, lines)
                    cloc = ';'.join(cloc)

                if('link_libs:' in lines[i]):
                    link_libs = parseSection(i, lines)

                if('link_libs_debug' in lines[i]):
                    link_libs_debug = parseSection(i, lines)


                if('link_libs_release' in lines[i]):
                    link_libs_release = parseSection(i, lines)

            for lib in link_libs:
                if lib in link_libs_debug:
                    link_libs_debug.remove(lib)
                if lib in link_libs_release:
                    link_libs_release.remove(lib)

            for lib in link_libs_debug:
                if lib in link_libs_release:
                    #print('removing duplicate {}'.format(lib))
                    link_libs.append(lib)
                    link_libs_debug.remove(lib)
                    link_libs_release.remove(lib)

            for i in range(len(link_libs)):
                base, name = splitLib(link_libs[i])
                link_libs[i] = name
                if base:
                    debug_dirs.append(base)
                    release_dirs.append(base)

            for i in range(len(link_libs_debug)):
                base, name = splitLib(link_libs_debug[i])
                link_libs_debug[i] = name
                if base:
                    debug_dirs.append(base)

            for i in range(len(link_libs_release)):
                base, name = splitLib(link_libs_release[i])
                link_libs_release[i] = name
                if base:
                    release_dirs.append(base)

            debug_dirs = removeDuplicates(debug_dirs)
            release_dirs = removeDuplicates(release_dirs)

            for x in link_libs:
                link_out.write('RUNTIME_COMPILER_LINKLIBRARY(LINK_PREFIX \"{}\" LINK_POSTFIX);\n'.format(x))
            j = link_file_lines.index('DEBUG_LIBS')
            link_out.write('\n'.join(link_file_lines[:j]))
            link_file_lines = link_file_lines[j+1:]

            for x in link_libs_debug:
                link_out.write('RUNTIME_COMPILER_LINKLIBRARY(LINK_PREFIX \"{}\" LINK_POSTFIX);\n'.format(x))
            j = link_file_lines.index('RELEASE_LIBS')
            link_out.write('\n'.join(link_file_lines[:j]))
            link_file_lines = link_file_lines[j+1:]

            for x in link_libs_release:
                link_out.write('RUNTIME_COMPILER_LINKLIBRARY(LINK_PREFIX \"{}\" LINK_POSTFIX);\n'.format(x))
            link_out.write('\n'.join(link_file_lines))

            out.write('#include "{}_export.hpp"\n\n'.format(args.plugin_name))

            out.write('namespace {}\n'.format(args.plugin_name))
            out.write('{\n')

            out.write('const char** getPluginIncludes()\n')
            writeList(out, include_dirs)
            out.write('}\n\n')

            out.write('const char** getPluginLinkDirsDebug()\n')
            writeList(out, debug_dirs)
            out.write('}\n\n')

            out.write('const char** getPluginLinkDirsRelease()\n')
            writeList(out, release_dirs)
            out.write('}\n\n')

            out.write('const char** getPluginCompileOptions()\n')
            tmp = []
            for option in compile_opts:
                tmp += option.split(' ')
            tmp = [x.strip() for x in tmp]
            compile_opts = []
            for x in tmp:
                if x not in compile_opts and 'no-undefined' not in x:
                    compile_opts.append(x)
            writeList(out, compile_opts, postfix=' ')
            out.write('}\n\n')

            out.write('const char** getPluginCompileDefinitions()\n')
            writeList(out, compile_defs + ['{}_EXPORTS'.format(args.plugin_name)], prefix='-D', postfix=' ')
            out.write('}\n\n')

            out.write('const char** getPluginLinkLibs()\n')
            writeList(out, link_libs, prefix='-l', postfix=' ')
            out.write('}\n\n')

            out.write('const char** getPluginLinkLibsDebug()\n')
            writeList(out, link_libs_debug, prefix='-l', postfix=' ')
            out.write('}\n\n')

            out.write('const char** getPluginLinkLibsRelease()\n')
            writeList(out, link_libs_release, prefix='-l', postfix=' ')
            out.write('}\n\n')

            out.write('int getPluginProjectId()\n')
            out.write('{\n')
            out.write('    return {};\n'.format(project_id.strip()))
            out.write('}\n\n')

            out.write('const char* getCompiler()\n')
            out.write('{\n')
            out.write('    return \"{}\";\n'.format(cloc))
            out.write('}\n')

            out.write('} // namespace ')
            out.write('{}\n'.format(args.plugin_name))
