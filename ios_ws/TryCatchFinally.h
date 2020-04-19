//
//  TryCatchFinally.h
//  TryCatchFinally
//
//  Modified by Bill LUO on 1/7/15.
//  Copyright (c) 2015 Understudy. All rights reserved.
//

#import <Foundation/Foundation.h>


@interface SwiftTryCatch : NSObject


/**
 Provides try catch functionality for swift by wrapping around Objective-C
 */
+ (void)tryBlock:(void(^)())tryBlock catchBlock:(void(^)(NSException*exception))catchBlock finallyBlock:(void(^)())finallyBlock;
+ (void)throwString:(NSString*)s;
+ (void)throwException:(NSException*)e;
@end
