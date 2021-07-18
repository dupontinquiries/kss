### deleted funcs
    def writeVideoGrouper(self, i, g, tmpChunks, returnChunksList, tmpName, folderName, referenceVideos, vidList):
        clamp = min( i + g, len(tmpChunks) - 1 )
        if i < clamp:
            print(f"({tmpName[:9]}) writeVideoGrouper {i} - {clamp}")
            if len(tmpChunks[i:clamp]) > 0:
                _v = list()
                k = i
                tmpMpyVideos = dict()
                while k < clamp:
                    if tmpChunks[k][0]:
                        j = k
                        while tmpChunks[k][0] and tmpChunks[k][1] == tmpChunks[j][1] and k < len(tmpChunks) - 1:
                            k += 1
                        video = None
                        #try:
                        if tmpChunks[k][1] not in tmpMpyVideos:
                            video = vidList[tmpChunks[k][1]].getFullVideo()
                            tmpMpyVideos[tmpChunks[k][1]] = video
                        else:
                            video = tmpMpyVideos[tmpChunks[k][1]]
                        #video = referenceVideos[ tmpChunks[k][1] ]
                        #except:
                        #    video = vidList[tmpChunks[k][1]].getFullVideo()
                        a = max(0, tmpChunks[j][4] / 1000)
                        b = min( video.duration, tmpChunks[k][5] / 1000 )
                        fps = video.fps
                        _v.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) )
                    k += 1
                if len(_v) == 0:
                    return
                outputMovie = None
                if len(_v) == 1:
                    outputMovie = _v
                elif len(_v) > 1:
                    outputMovie = mpye.concatenate_videoclips(_v, method='compose')
                clip = outD.append(f'{folderName}_threading').append(f'{i//g} - {tmpName}_tmpChunk.mp4')
                outputMovie.write_videofile(clip.aPath(), preset='slow', threads=16) #, codec='libx265', audio_codec='aac', audio_bitrate='48k', threads=16) #12
                returnChunksList.append((i, clip))


    def writeVideo_multithreading(self, tmpChunks, referenceVideos, vidList):
        print(f"begin (main) writeVideo()")
        import threading
        threads = list()
        returnChunksList = list()
        nConcurrentThreads=1+(len(tmpChunks)//self.chulenms//40)
        sparsity = len(tmpChunks)//nConcurrentThreads
        folderName = randomString(9)

        count = 0
        for i in range( 0, len(tmpChunks), sparsity ):

            if len(threads) > 10:
                threads[0].join()
                threads.pop(0)

            t = threading.Thread(target=self.writeVideoGrouper, args=(i, sparsity, tmpChunks, returnChunksList, randomString(9), folderName, referenceVideos, vidList, ))
            threads.append(t)
            t.start()
            count += 1

        for thread in threads:
            thread.join()

        videoList = list()
        for i, path in sorted(returnChunksList):
            videoList.append(path.getFullVideo())


        outputMovie = mpye.concatenate_videoclips(videoList)
        outputMovie.write_videofile(outD.append(f'{randomString(5)} -- output.mp4').aPath(), preset='slow', codec='libx265', audio_codec='aac', audio_bitrate='48k') #, threads=16)

        for i, path in sorted(returnChunksList):
            if False:
                path.delete()

        cleanup = False
        if cleanup:
            executor = concurrent.futures.ProcessPoolExecutor(6)
            futures = [executor.submit(tmpChunks[i].delete())
               for i in range( len(tmpChunks) )]
            concurrent.futures.wait(futures)
        print(f"  end (main) writeVideo()")


    def writeVideoGrouper_multiprocessing(self, queue, i, g, tmpChunks, tmpName, folderName, vidList):
        clamp = min( i + g, len(tmpChunks) - 1 )
        if i < clamp:
            print(f"({tmpName[:9]}) writeVideoGrouper {i} - {clamp}")
            if len(tmpChunks[i:clamp]) > 0:
                _v = list()
                k = i
                tmpMpyVideos = dict()
                while k < clamp:
                    if tmpChunks[k][0]:
                        j = k
                        while tmpChunks[k][0] and tmpChunks[k][1] == tmpChunks[j][1] and k < len(tmpChunks) - 1:
                            k += 1
                        video = None
                        #try:
                        if tmpChunks[k][1] not in tmpMpyVideos:
                            video = vidList[tmpChunks[k][1]].getFullVideo()
                            tmpMpyVideos[tmpChunks[k][1]] = video
                        else:
                            video = tmpMpyVideos[tmpChunks[k][1]]
                        a = max(0, tmpChunks[j][4] / 1000)
                        b = min( video.duration, tmpChunks[k][5] / 1000 )
                        fps = video.fps
                        _v.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) )
                    k += 1
                if len(_v) == 0:
                    return
                outputMovie = None
                if len(_v) == 1:
                    outputMovie = _v
                elif len(_v) > 1:
                    outputMovie = mpye.concatenate_videoclips(_v, method='compose')
                clip = outD.append(f'{folderName}_threading').append(f'{i//g} - {tmpName}_tmpChunk.mp4')
                outputMovie.write_videofile(clip.aPath(), preset='slow', threads=16)
                queue.put( (i, clip) )


    def writeVideo_multiprocessing(self, tmpChunks, vidList):
        print(f"begin (main) writeVideo()")
        from multiprocessing import Process, Queue
        queue = Queue()
        nConcurrentThreads=1+(len(tmpChunks)//self.chulenms//25)
        sparsity = len(tmpChunks)//nConcurrentThreads
        folderName = randomString(9)

        processes = [ Process(  target=self.writeVideoGrouper_multiprocessing, args=(queue, i, sparsity, tmpChunks, randomString(9), folderName, vidList, )  ) for i in range( 0, len(tmpChunks), sparsity ) ]

        count = 0
        for p in processes:
            p.start()
            ++count

        for p in processes:
            p.join()

        results = [queue.get() for p in processes]

        videoList = list()
        for i, path in sorted(results):
            videoList.append(path.getFullVideo())

        outputMovie = mpye.concatenate_videoclips(videoList)
        outputMovie.write_videofile(outD.append(f'{randomString(5)} -- output.mp4').aPath(), preset='slow', codec='libx265', audio_codec='aac', audio_bitrate='48k') #, threads=16)

        cleanup = False
        if cleanup:
            executor = concurrent.futures.ProcessPoolExecutor(6)
            futures = [executor.submit(tmpChunks[i].delete())
               for i in range( len(tmpChunks) )]
            concurrent.futures.wait(futures)
        print(f"  end (main) writeVideo()")
